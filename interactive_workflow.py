"""Guarded ipywidgets presentation for the bounded scientific controller."""

from __future__ import annotations

from math import isclose, isfinite
from html import escape
from typing import Any

import ipywidgets as widgets
from IPython.display import display as ipython_display
from pydantic import BaseModel, ValidationError

import demo_agent
from command_receipts import CommandReceipt, command_receipt
from objective_challenge import (
    MAX_ATTEMPTS, SCORE_TOLERANCE, ObjectiveAttempt, ObjectiveSwap,
    objective_figures,
)
from objective_receipts import objective_receipt


_MODELS = demo_agent.TOOL_ARGUMENT_MODELS


def controls_for(proposal: demo_agent.StageProposal) -> dict[str, widgets.Widget]:
    """Return only schema-encoding controls for an exact validated proposal."""
    if proposal.stage not in _MODELS or type(proposal.arguments) is not _MODELS[proposal.stage]:
        raise ValueError("Proposal stage and argument model must match.")
    arguments = proposal.arguments
    if proposal.stage in {"inspect_library", "measure_tanimoto_similarity", "optimize_conformers_mmff94"}:
        return {}
    if proposal.stage == "generate_morgan_fingerprints":
        return {
            "radius": widgets.Dropdown(options=(2, 3), value=arguments.radius, description="Radius"),
            "size": widgets.Dropdown(options=(1024, 2048), value=arguments.size, description="Size"),
        }
    if proposal.stage == "discover_fused_butina_clusters":
        return {"cutoff": widgets.FloatSlider(
            value=arguments.cutoff, min=0.40, max=0.60, step=0.05,
            readout_format=".2f", description="Cutoff", continuous_update=False,
        )}
    if proposal.stage == "embed_representative_conformers":
        return {
            "representative_count": widgets.IntSlider(
                value=arguments.representative_count, min=3, max=6, step=1,
                description="Representatives", continuous_update=False,
            ),
            "policy": widgets.Dropdown(
                options=("largest_clusters_first", "include_singleton_if_available"),
                value=arguments.policy, description="Policy",
            ),
            "conformers_per_representative": widgets.IntSlider(
                value=arguments.conformers_per_representative, min=3, max=8, step=1,
                description="Conformers", continuous_update=False,
            ),
        }
    raise ValueError("Unsupported workflow stage.")


def _approved_model(proposal: demo_agent.StageProposal, controls: dict[str, widgets.Widget]) -> BaseModel:
    model = _MODELS.get(proposal.stage)
    if model is None or type(proposal.arguments) is not model:
        raise ValueError("Proposal stage and argument model must match.")
    values = {name: control.value for name, control in controls.items()}
    if "decision_basis" in model.model_fields:
        values["decision_basis"] = proposal.arguments.decision_basis
    return model.model_validate(values)


def _parameters(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude={"decision_basis"})


def _safe_message(error: Exception) -> str:
    if isinstance(error, demo_agent.ToolCallError):
        return escape(str(error))
    if str(error) == demo_agent.AUTH_GUIDANCE:
        return escape(demo_agent.AUTH_GUIDANCE)
    return "A local workflow error occurred. The workflow was stopped safely."


class InteractiveWorkflow:
    """One guarded, linear stage-card view over ``BoundedWorkflowController``."""

    def __init__(self, controller: demo_agent.BoundedWorkflowController):
        self.controller = controller
        self.status = "idle"
        self.controls: dict[str, widgets.Widget] = {}
        self.approve_button: widgets.Button | None = None
        self.retry_button: widgets.Button | None = None
        self.transcript_text = ""
        self.plan_cards: tuple[widgets.VBox, ...] = ()
        self.completed_cards: tuple[widgets.VBox, ...] = ()
        self.completed_results: tuple[Any, ...] = ()
        self.active_card: widgets.VBox | None = None
        self.objective_card: widgets.VBox | None = None
        self.objective_button: widgets.Button | None = None
        self.objective_summary = widgets.HTML()
        self.objective_attempt_cards: tuple[widgets.Accordion, ...] = ()
        self.objective_attempt_box = widgets.VBox()
        self.objective_output = widgets.Output()
        self.workflow_result: Any | None = None
        self._active_proposal: demo_agent.StageProposal | None = None
        self._approved: BaseModel | None = None
        self._control_observers: list[tuple[widgets.Widget, Any]] = []
        self._busy = False
        self._objective_retry_used = False
        self.start_button = widgets.Button(description="Start Agent", button_style="primary")
        self.start_button.on_click(self.start)
        self._body = widgets.VBox()
        self.root = widgets.VBox((self.start_button, self._body))

    def display(self) -> widgets.VBox:
        ipython_display(self.root)
        return self.root

    def _line(self, text: str) -> None:
        self.transcript_text += text + "\n"

    def _set_body(self) -> None:
        children = [*self.plan_cards, *self.completed_cards]
        if self.objective_card is not None:
            children.append(self.objective_card)
        if self.active_card is not None:
            children.append(self.active_card)
        self._body.children = tuple(children)

    @staticmethod
    def _known_failure(error: Exception) -> bool:
        return isinstance(error, demo_agent.ToolCallError) or str(error) == demo_agent.AUTH_GUIDANCE

    def _retry_card(self, title: str, error: Exception, status: str, label: str, callback) -> None:
        message = _safe_message(error)
        self.status = status
        self._line(f"{title}: {message}")
        button = widgets.Button(description=label)
        button.on_click(callback)
        self.retry_button = button
        error_widget = widgets.HTML(f"<b>{escape(title)}</b><p>{message}</p>")
        if self.active_card is None or status in {"plan_failed", "proposal_failed"}:
            self.active_card = widgets.VBox((error_widget, button))
        else:
            self.active_card.children = (*self.active_card.children, error_widget, button)
        self._set_body()

    def _stop(self) -> None:
        self._detach_observers()
        self.status = "stopped"
        if self.retry_button is not None:
            self.retry_button.disabled = True
        if self.approve_button is not None:
            self.approve_button.disabled = True
        for control in self.controls.values():
            control.disabled = True
        self.retry_button = None
        self.approve_button = None
        message = "A local workflow error occurred. The workflow was stopped safely."
        self._line(f"Workflow stopped: {message}")
        error_widget = widgets.HTML(f"<b>Workflow stopped</b><p>{message}</p>")
        if self.active_card is None:
            self.active_card = widgets.VBox((error_widget,))
        else:
            self.active_card.children = (*self.active_card.children, error_widget)
        self._set_body()

    def _detach_observers(self) -> None:
        for control, callback in self._control_observers:
            try:
                control.unobserve(callback, names="value")
            except Exception:
                pass
        self._control_observers = []

    def _plan_retryable(self) -> bool:
        try:
            return (
                self.controller.plan is None
                and self.controller.pending is None
                and self.controller.session.turn_count == 0
                and self.controller.session.state.phase is demo_agent.WorkflowPhase.NEW
                and self.controller.stage_results == []
                and self.controller.report is None
                and self.controller.synthesis_prompt_appended is False
            )
        except Exception:
            return False

    def _synthesis_retryable(self) -> bool:
        try:
            return (
                len(self.completed_cards) == len(demo_agent.STAGES)
                and self.controller.plan is not None
                and self.controller.pending is None
                and self.controller.session.state.phase is demo_agent.WorkflowPhase.OPTIMIZED
                and tuple(result.stage for result in self.controller.stage_results) == demo_agent.STAGES
                and self.controller.report is not None
                and self.controller.synthesis_prompt_appended is True
                and self.controller.objective_run is not None
                and self.controller.objective_evidence is not None
                and self.controller.pending_objective is None
                and self.controller.objective_prompt_appended is True
                and 7
                <= self.controller.session.turn_count
                <= demo_agent.MAX_OBJECTIVE_HOSTED_TURNS
            )
        except Exception:
            return False

    def _objective_retryable(self) -> bool:
        try:
            attempts = tuple(self.controller.objective_attempts)
            return (
                not self._objective_retry_used
                and len(self.completed_cards) == len(demo_agent.STAGES)
                and self.controller.plan is not None
                and self.controller.pending is None
                and self.controller.session.state.phase is demo_agent.WorkflowPhase.OPTIMIZED
                and tuple(result.stage for result in self.controller.stage_results)
                == demo_agent.STAGES
                and self.controller.report is not None
                and self.controller.objective_context is not None
                and self.controller.objective_prompt_appended is True
                and self.controller.objective_run is None
                and self.controller.objective_evidence is None
                and self.controller.pending_objective is None
                and self.controller.pending_objective_swap is None
                and self.controller.objective_rejection_count
                < demo_agent.MAX_OBJECTIVE_CORRECTIONS
                and len(attempts) < MAX_ATTEMPTS
                and (not attempts or bool(self.controller.objective_suggestions))
                and tuple(attempt.attempt_number for attempt in attempts)
                == tuple(range(1, len(attempts) + 1))
                and self.controller.session.turn_count
                == 7 + len(attempts) + self.controller.objective_rejection_count
            )
        except Exception:
            return False

    def start(self, button: widgets.Button | None = None) -> None:
        if (button is not None and button is not self.start_button) or self._busy or self.status != "idle" or self.start_button.disabled:
            return
        self._busy = True
        self.start_button.disabled = True
        self.status = "planning"
        try:
            plan = self.controller.request_plan()
            lines = "".join(
                f"<li><code>{escape(item.stage)}</code> — {escape(item.rationale)}</li>"
                for item in plan.stages
            )
            self._line("Plan: " + ", ".join(item.stage for item in plan.stages))
            plan_card = widgets.VBox((widgets.HTML(f"<h3>Fixed workflow plan</h3><ol>{lines}</ol>"),))
            self.plan_cards = (*self.plan_cards, plan_card)
            self.active_card = None
            self._set_body()
            self._request_proposal()
        except Exception as error:
            if self._known_failure(error) and self._plan_retryable():
                self._retry_card("Plan request failed", error, "plan_failed", "Retry Plan", self._retry_plan)
            else:
                self._stop()
        finally:
            self._busy = False

    def _retry_plan(self, button: widgets.Button) -> None:
        if self._busy or self.status != "plan_failed" or button is not self.retry_button or button.disabled:
            return
        if not self._plan_retryable():
            button.disabled = True
            self._stop()
            return
        button.disabled = True
        self.retry_button = None
        self.status = "idle"
        self.start_button.disabled = False
        self.start(self.start_button)

    def _evidence_summary(self) -> str:
        results = getattr(self.controller, "stage_results", ())
        if not results:
            return "No prior scientific results; fixed input and plan only."
        result = results[-1]
        allowed = demo_agent._STAGE_METRICS.get(result.stage, ())
        metrics = [f"{key}={result.summary[key]}" for key in allowed if key in result.summary]
        return f"{result.stage}: " + (", ".join(metrics) if metrics else "completed")

    def _request_proposal(self) -> None:
        self.status = "proposing"
        try:
            proposal = self.controller.request_next_stage()
            self._show_proposal(proposal)
        except Exception as error:
            expected = demo_agent.STAGES[len(self.completed_cards)] if len(self.completed_cards) < len(demo_agent.STAGES) else None
            retryable = False
            try:
                retryable = getattr(self.controller, "pending", None) is None and self.controller.session.eligible_tool_name() == expected
            except Exception:
                pass
            if self._known_failure(error) and retryable:
                self._retry_card("Stage proposal failed", error, "proposal_failed", "Retry Proposal", self._retry_proposal)
            else:
                self._stop()

    def _retry_proposal(self, button: widgets.Button) -> None:
        if self._busy or self.status != "proposal_failed" or button is not self.retry_button or button.disabled:
            return
        expected = demo_agent.STAGES[len(self.completed_cards)]
        try:
            safe = getattr(self.controller, "pending", None) is None and self.controller.session.eligible_tool_name() == expected
        except Exception:
            safe = False
        if not safe:
            button.disabled = True
            self._stop()
            return
        self._busy = True
        button.disabled = True
        self.retry_button = None
        try:
            self._request_proposal()
        finally:
            self._busy = False

    def _show_proposal(self, proposal: demo_agent.StageProposal) -> None:
        self._active_proposal = proposal
        self._approved = None
        self.retry_button = None
        card_controls = controls_for(proposal)
        self.controls = card_controls
        self._control_observers = []
        proposed_receipt = command_receipt(proposal.stage, proposal.arguments)
        preview = widgets.HTML()

        def update_preview(_change=None):
            try:
                approved = _approved_model(proposal, card_controls)
                preview.value = "<b>Approved-call preview</b><pre>" + escape(
                    command_receipt(proposal.stage, approved).approved_tool_call
                ) + "</pre>"
            except (ValidationError, ValueError):
                preview.value = "<b>Approved-call preview unavailable</b>"
            except Exception:
                self._stop()

        for control in card_controls.values():
            control.observe(update_preview, names="value")
            self._control_observers.append((control, update_preview))
        update_preview()
        if self.status == "stopped":
            return
        button = widgets.Button(description="Approve & Run", button_style="success")
        button.on_click(self._approve)
        self.approve_button = button
        basis = getattr(proposal.arguments, "decision_basis", "Fixed parameter-free stage.")
        header = widgets.HTML(
            f"<h3>{escape(proposal.stage)}</h3>"
            f"<p><b>Available evidence:</b> {escape(self._evidence_summary())}</p>"
            f"<p><b>Decision summary:</b> {escape(basis)}</p>"
            f"<b>Proposed tool call</b><pre>{escape(proposed_receipt.approved_tool_call)}</pre>"
        )
        self.active_card = widgets.VBox((header, *self.controls.values(), preview, button))
        self.status = "awaiting_approval"
        self._line(f"Proposal {proposal.stage}: {proposed_receipt.approved_tool_call}")
        self._set_body()

    def _disable_active(self) -> None:
        if self.approve_button is not None:
            self.approve_button.disabled = True
        for control in self.controls.values():
            control.disabled = True

    def _approve(self, original_button: widgets.Button) -> None:
        if self._busy or original_button is not self.approve_button or original_button.disabled or self.status != "awaiting_approval":
            return
        self._busy = True
        self.status = "executing"
        self._disable_active()
        proposal = self._active_proposal
        try:
            assert proposal is not None
            approved = _approved_model(proposal, self.controls)
            self._approved = approved
            receipt = command_receipt(proposal.stage, approved)
            result = self.controller.execute_pending(approved)
            self._complete_card(proposal, approved, receipt, result)
            if len(self.completed_cards) == len(demo_agent.STAGES):
                self._show_objective_challenge()
            else:
                self._request_proposal()
        except Exception as error:
            if self._known_failure(error) and self._execution_is_retryable(proposal):
                self._retry_card("Scientific execution failed", error, "execution_failed", "Retry Execution", self._retry_execution)
            else:
                self._stop()
        finally:
            self._busy = False

    def _execution_is_retryable(self, proposal) -> bool:
        if proposal is None or self._approved is None:
            return False
        try:
            return self.controller.pending is proposal and self.controller.session.eligible_tool_name() == proposal.stage
        except Exception:
            return False

    def _retry_execution(self, button: widgets.Button) -> None:
        if self._busy or self.status != "execution_failed" or button is not self.retry_button or button.disabled:
            return
        proposal = self._active_proposal
        if not self._execution_is_retryable(proposal):
            button.disabled = True
            self._stop()
            return
        self._busy = True
        self.status = "executing"
        button.disabled = True
        self.retry_button = None
        approved = self._approved
        try:
            assert proposal is not None and approved is not None
            receipt = command_receipt(proposal.stage, approved)
            result = self.controller.execute_pending(approved)
            self._complete_card(proposal, approved, receipt, result)
            if len(self.completed_cards) == len(demo_agent.STAGES):
                self._show_objective_challenge()
            else:
                self._request_proposal()
        except Exception as error:
            if self._known_failure(error) and self._execution_is_retryable(proposal):
                self._retry_card("Scientific execution failed", error, "execution_failed", "Retry Execution", self._retry_execution)
            else:
                self._stop()
        finally:
            self._busy = False

    def _complete_card(self, proposal, approved, receipt: CommandReceipt, result) -> None:
        proposed_values, approved_values = _parameters(proposal.arguments), _parameters(approved)
        changed = proposed_values != approved_values
        comparison = f"<p><b>Proposed:</b> {escape(str(proposed_values))}<br><b>Approved:</b> {escape(str(approved_values))}</p>" if changed else "<p>Proposal approved unchanged.</p>"
        metrics = {key: result.summary[key] for key in demo_agent._STAGE_METRICS[result.stage] if key in result.summary}
        output = widgets.Output()
        completion = widgets.HTML(
            f"<h3>Completed: {escape(result.stage)}</h3>{comparison}"
            f"<b>Approved tool call</b><pre>{escape(receipt.approved_tool_call)}</pre>"
            f"<b>{escape(receipt.scientific_label)}</b><pre>{escape(receipt.scientific_invocation)}</pre>"
            f"<p><b>Result metrics:</b> {escape(str(metrics))}</p>"
        )
        card = self.active_card
        if card is None:
            raise RuntimeError("Active proposal card was missing.")
        card.children = (*card.children, completion, output)
        self._detach_observers()
        self.completed_cards = (*self.completed_cards, card)
        self.completed_results = (*self.completed_results, result)
        self.active_card = None
        self.status = "proposing"
        self._line(f"Completed {result.stage}: {receipt.approved_tool_call}; {receipt.scientific_label}: {receipt.scientific_invocation}; metrics={metrics}")
        self._approved = None
        self._active_proposal = None
        self.retry_button = None
        self.approve_button = None
        self._set_body()
        for figure in result.figures:
            try:
                with output:
                    demo_agent._display_figure(figure)
            except Exception:
                placeholder = "Figure unavailable in this notebook frontend."
                card.children = (*card.children, widgets.HTML(f"<p>{placeholder}</p>"))
                self._line(placeholder)

    @staticmethod
    def _objective_summary_html(context, attempts, run=None) -> str:
        display_values = tuple(
            InteractiveWorkflow._objective_display_values(
                attempt.score,
                context.target_score,
                (
                    attempt.score - context.baseline_score
                    if attempt.selected_swap is None
                    else attempt.selected_swap.score_delta
                ),
            )
            for attempt in attempts
        )
        precision = max((values[3] for values in display_values), default=3)
        scientific = any(values[4] for values in display_values)
        score_items = [
            (
                "Baseline",
                context.baseline_score,
                "Current largest-clusters-first policy",
            )
        ]
        score_items.extend(
            (
                f"Attempt {attempt.attempt_number}",
                attempt.score,
                "Goal achieved" if attempt.achieved else "Revise",
            )
            for attempt in attempts
        )
        score_strip = "".join(
            (
                "<span style='display:inline-block;padding:6px 10px;margin:2px;"
                "border:1px solid #aaa;border-radius:12px'>"
                f"<b>{escape(label)}</b> "
                f"{InteractiveWorkflow._objective_scalar(score, precision, scientific)}<br>"
                f"<small>{escape(status)}</small></span>"
            )
            for label, score, status in score_items
        )
        decision_ladder = "".join(
            InteractiveWorkflow._objective_attempt_row(
                context,
                attempt,
                attempts[index - 1] if index else None,
            )
            for index, attempt in enumerate(attempts)
        )
        final = ""
        if run is not None:
            label = (
                "Goal achieved"
                if run.achieved
                else "Objective not achieved within attempt limit"
            )
            if run.termination_reason == "baseline_already_optimal":
                label = "Baseline already optimal within the bounded pool"
            final = f"<p><b>Outcome:</b> {escape(label)}</p>"
        return (
            "<p><b>Objective:</b> Select four MMFF94-parameter-eligible compounds "
            "that maximize minimum pairwise Morgan/Tanimoto distance.</p>"
            "<p><b>Constraints:</b> four unique supplied IDs · four distinct fused "
            "Butina clusters · fixed fingerprint evidence · at most three attempts</p>"
            "<p><b>Target:</b> D_min ≥ "
            f"{InteractiveWorkflow._objective_scalar(context.target_score, precision, scientific)} "
            "(80% of attainable improvement over baseline)</p>"
            f"<div>{score_strip}</div>"
            f"<div aria-label='Objective decision ladder'>{decision_ladder}</div>{final}"
        )

    @staticmethod
    def _objective_attempt_row(context, attempt, prior_attempt=None) -> str:
        """Render one measured decision step without exposing model rationale."""
        if (
            not InteractiveWorkflow._is_finite_float(context.baseline_score)
            or not InteractiveWorkflow._is_finite_float(context.target_score)
        ):
            raise ValueError("Objective attempt rows require finite context scores.")
        if type(attempt) is not ObjectiveAttempt:
            raise ValueError("Objective attempt rows require exact objective attempts.")
        if type(attempt.attempt_number) is not int or not 1 <= attempt.attempt_number <= MAX_ATTEMPTS:
            raise ValueError("Objective attempt rows require bounded integer attempt numbers.")
        if not InteractiveWorkflow._is_canonical_panel(attempt.selected_ids):
            raise ValueError("Objective attempt rows require canonical selected panels.")
        if not InteractiveWorkflow._is_finite_float(attempt.score):
            raise ValueError("Objective attempt rows require finite measured scores.")
        if (
            attempt.constraints_passed is not True
            or type(attempt.achieved) is not bool
            or attempt.achieved != (
                attempt.score + SCORE_TOLERANCE >= context.target_score
            )
            or not InteractiveWorkflow._is_canonical_pair(
                attempt.limiting_pair, attempt.selected_ids
            )
        ):
            raise ValueError("Objective attempt rows require measured objective truth.")
        swap = attempt.selected_swap
        if attempt.attempt_number == 1:
            if swap is not None or prior_attempt is not None:
                raise ValueError("The initial objective attempt cannot carry revision state.")
            action = f"Initial panel — {attempt.decision_basis}"
            delta = attempt.score - context.baseline_score
            accent = "#76B900" if attempt.achieved else "#6c757d"
        else:
            if type(swap) is not ObjectiveSwap:
                raise ValueError("A revision requires an exact selected swap.")
            if type(prior_attempt) is not ObjectiveAttempt:
                raise ValueError("A revision requires its exact prior attempt.")
            if prior_attempt.attempt_number != attempt.attempt_number - 1:
                raise ValueError("A revision requires its consecutive prior attempt.")
            if (
                not InteractiveWorkflow._is_canonical_panel(prior_attempt.selected_ids)
                or not InteractiveWorkflow._is_finite_float(prior_attempt.score)
                or prior_attempt.constraints_passed is not True
                or prior_attempt.achieved is not False
                or not InteractiveWorkflow._is_canonical_pair(
                    prior_attempt.limiting_pair, prior_attempt.selected_ids
                )
                or not InteractiveWorkflow._is_canonical_panel(swap.resulting_ids)
                or not InteractiveWorkflow._is_canonical_pair(
                    swap.limiting_pair, swap.resulting_ids
                )
                or not InteractiveWorkflow._is_finite_float(swap.predicted_score)
                or not InteractiveWorkflow._is_finite_float(swap.score_delta)
                or swap.score_delta <= SCORE_TOLERANCE
                or not InteractiveWorkflow._same_panel(swap.resulting_ids, attempt.selected_ids)
                or swap.replacement_id not in swap.resulting_ids
                or swap.replace_id in swap.resulting_ids
            ):
                raise ValueError("A revision has invalid selected-swap provenance.")
            replacement_position = swap.resulting_ids.index(swap.replacement_id)
            predecessor = (
                swap.resulting_ids[:replacement_position]
                + (swap.replace_id,)
                + swap.resulting_ids[replacement_position + 1 :]
            )
            if (
                not InteractiveWorkflow._same_panel(predecessor, prior_attempt.selected_ids)
                or not isclose(attempt.score, swap.predicted_score, rel_tol=0.0, abs_tol=1e-12)
                or attempt.limiting_pair != swap.limiting_pair
                or not isclose(
                    swap.score_delta,
                    attempt.score - prior_attempt.score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("A revision does not match its prior measurement.")
            action = (
                f"replace {swap.replace_id} with {swap.replacement_id} — "
                f"{attempt.decision_basis}"
            )
            delta = swap.score_delta
            accent = "#76B900" if attempt.achieved else "#D68A00"
        score_text, target_text, delta_text, precision, scientific = (
            InteractiveWorkflow._objective_display_values(
                attempt.score, context.target_score, delta
            )
        )
        if attempt.attempt_number == 1:
            observation = (
                "baseline D_min "
                f"{InteractiveWorkflow._objective_scalar(context.baseline_score, precision, scientific)}; "
                f"current D_min {score_text}; initial agent proposal"
            )
        else:
            observation = (
                "prior D_min "
                f"{InteractiveWorkflow._objective_scalar(prior_attempt.score, precision, scientific)}; "
                f"limiting pair {prior_attempt.limiting_pair[0]} / {prior_attempt.limiting_pair[1]}"
            )
        if attempt.achieved and attempt.score < context.target_score:
            comparison = (
                f"{score_text} meets target within 1e-12 tolerance "
                f"(score + 1e-12 ≥ {target_text})"
            )
        elif attempt.achieved:
            comparison = f"{score_text} ≥ {target_text}"
        else:
            comparison = f"{score_text} < {target_text}"
        outcome = "Goal achieved" if attempt.achieved else "Revise"
        return (
            f"<section style='border-left:3px solid {accent};padding:6px 10px;"
            "margin:6px 0' aria-label='Objective attempt'>"
            f"<b>Attempt {attempt.attempt_number}</b> · {escape(outcome)}<br>"
            f"<small><b>Observe:</b> {escape(observation)}</small><br>"
            f"<small><b>Agent action:</b> {escape(action)}</small><br>"
            f"<small><b>Measure:</b> D_min {score_text} · Δ {delta_text} · {escape(comparison)}</small><br>"
            f"<small><b>Outcome:</b> {escape(outcome)}</small>"
            "</section>"
        )

    @staticmethod
    def _objective_precision(score: float, target: float, delta: float) -> int:
        """Use the shortest honest fixed precision for one decision measurement."""
        for precision in range(3, 7):
            score_text = f"{score:.{precision}f}"
            target_text = f"{target:.{precision}f}"
            delta_text = f"{delta:+.{precision}f}"
            indistinguishable_scores = score != target and score_text == target_text
            erased_delta = delta != 0.0 and float(delta_text) == 0.0
            if not indistinguishable_scores and not erased_delta:
                return precision
        return 6

    @staticmethod
    def _objective_display_values(
        score: float, target: float, delta: float
    ) -> tuple[str, str, str, int, bool]:
        """Format one comparison without rounding a true difference away."""
        precision = InteractiveWorkflow._objective_precision(score, target, delta)
        comparison_scientific = (
            score != target
            and f"{score:.{precision}f}" == f"{target:.{precision}f}"
        )
        delta_scientific = delta != 0.0 and float(f"{delta:+.{precision}f}") == 0.0
        if comparison_scientific:
            scientific_precision = InteractiveWorkflow._scientific_precision(
                score, target
            )
            return (
                f"{score:.{scientific_precision}e}",
                f"{target:.{scientific_precision}e}",
                (
                    f"{delta:+.{scientific_precision}e}"
                    if delta_scientific
                    else f"{delta:+.{precision}f}"
                ),
                scientific_precision,
                True,
            )
        if delta_scientific:
            return (
                f"{score:.{precision}f}",
                f"{target:.{precision}f}",
                f"{delta:+.0e}",
                precision,
                False,
            )
        return (
            f"{score:.{precision}f}",
            f"{target:.{precision}f}",
            f"{delta:+.{precision}f}",
            precision,
            False,
        )

    @staticmethod
    def _objective_scalar(value: float, precision: int, scientific: bool) -> str:
        return f"{value:.{precision}e}" if scientific else f"{value:.{precision}f}"

    @staticmethod
    def _scientific_precision(first: float, second: float) -> int:
        for precision in range(18):
            if f"{first:.{precision}e}" != f"{second:.{precision}e}":
                return precision
        return 17

    @staticmethod
    def _is_finite_float(value: object) -> bool:
        return type(value) is float and isfinite(value)

    @staticmethod
    def _is_canonical_panel(value: object) -> bool:
        return (
            type(value) is tuple
            and len(value) == 4
            and len(set(value)) == len(value)
            and all(type(molecule_id) is str for molecule_id in value)
        )

    @staticmethod
    def _is_canonical_pair(value: object, panel: tuple[str, ...]) -> bool:
        return (
            type(value) is tuple
            and len(value) == 2
            and all(type(molecule_id) is str and molecule_id for molecule_id in value)
            and value[0] != value[1]
            and all(molecule_id in panel for molecule_id in value)
        )

    @staticmethod
    def _same_panel(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
        return len(first) == len(second) and set(first) == set(second)

    def _show_objective_challenge(self) -> None:
        self.status = "objective_initializing"
        try:
            context = self.controller.begin_objective_challenge()
        except Exception as error:
            self._stop()
            return
        self.objective_summary = widgets.HTML(
            self._objective_summary_html(
                context,
                tuple(self.controller.objective_attempts),
                self.controller.objective_run,
            )
        )
        button = widgets.Button(
            description="Run Objective Challenge", button_style="success"
        )
        button.on_click(self._run_objective_challenge)
        self.objective_button = button
        self.objective_attempt_cards = ()
        self.objective_attempt_box = widgets.VBox()
        self.objective_output = widgets.Output()
        self.objective_card = widgets.VBox(
            (
                widgets.HTML("<h3>Objective-Driven Agent Challenge</h3>"),
                self.objective_summary,
                button,
                self.objective_attempt_box,
                self.objective_output,
            )
        )
        self.active_card = None
        self.status = "objective_ready"
        self._line("Objective challenge ready")
        self._set_body()
        if self.controller.objective_run is not None:
            button.disabled = True
            self._finish_objective_challenge()

    def _append_objective_attempt(self, proposal, attempt) -> None:
        prior_attempt = (
            self.controller.objective_attempts[-2]
            if attempt.attempt_number > 1
            else None
        )
        receipt = objective_receipt(proposal, attempt.selected_swap, prior_attempt)
        result_label = "Goal achieved" if attempt.achieved else "Revise"
        intervention = ""
        if receipt.validated_intervention is not None:
            intervention = (
                "<b>Validated intervention</b>"
                f"<pre>{escape(receipt.validated_intervention)}</pre>"
            )
        details = widgets.HTML(
            f"<p><b>Decision summary:</b> {escape(attempt.decision_basis)}</p>"
            "<b>Validated Nemotron proposal</b>"
            f"<pre>{escape(receipt.validated_proposal)}</pre>"
            f"{intervention}"
            "<b>Evaluation executed by Python</b>"
            f"<pre>{escape(receipt.python_evaluation)}</pre>"
            f"<p><b>D_min:</b> {attempt.score:.3f} &nbsp; "
            f"<b>Limiting pair:</b> {escape(' / '.join(attempt.limiting_pair))} "
            f"&nbsp; <b>Result:</b> {escape(result_label)}</p>"
        )
        for prior in self.objective_attempt_cards:
            prior.selected_index = None
        card = widgets.Accordion((details,))
        card.set_title(0, f"Attempt {attempt.attempt_number} — {result_label}")
        card.selected_index = 0
        self.objective_attempt_cards = (*self.objective_attempt_cards, card)
        self.objective_attempt_box.children = self.objective_attempt_cards
        self.objective_summary.value = self._objective_summary_html(
            self.controller.objective_context,
            tuple(self.controller.objective_attempts),
            self.controller.objective_run,
        )
        self._line(
            f"Objective attempt {attempt.attempt_number}: "
            f"score={attempt.score:.3f}; limiting_pair={attempt.limiting_pair}; "
            f"result={result_label}"
        )
        self._set_body()

    def _run_objective_challenge(self, button: widgets.Button) -> None:
        if (
            self._busy
            or self.status != "objective_ready"
            or button is not self.objective_button
            or button.disabled
        ):
            return
        self._busy = True
        button.disabled = True
        try:
            self._continue_objective_challenge()
        finally:
            self._busy = False

    def _continue_objective_challenge(self) -> None:
        self.status = "objective_running"
        try:
            while self.controller.objective_run is None:
                proposal = self.controller.request_objective_attempt()
                attempt = self.controller.execute_objective_attempt(proposal)
                self._append_objective_attempt(proposal, attempt)
            self._finish_objective_challenge()
        except Exception as error:
            if self._known_failure(error) and self._objective_retryable():
                self._retry_card(
                    "Objective proposal failed",
                    error,
                    "objective_failed",
                    "Retry Objective Proposal",
                    self._retry_objective,
                )
            else:
                self._stop()

    def _retry_objective(self, button: widgets.Button) -> None:
        if (
            self._busy
            or self.status != "objective_failed"
            or button is not self.retry_button
            or button.disabled
        ):
            return
        if not self._objective_retryable():
            button.disabled = True
            self._stop()
            return
        self._busy = True
        self._objective_retry_used = True
        button.disabled = True
        self.retry_button = None
        self.active_card = None
        self._set_body()
        try:
            self._continue_objective_challenge()
        finally:
            self._busy = False

    def _finish_objective_challenge(self) -> None:
        run = self.controller.objective_run
        if run is None:
            raise RuntimeError("Objective challenge did not terminate.")
        self.objective_summary.value = self._objective_summary_html(
            self.controller.objective_context,
            tuple(self.controller.objective_attempts),
            run,
        )
        try:
            figures = objective_figures(run, self.controller.session.state)
            with self.objective_output:
                for figure in figures:
                    demo_agent._display_figure(figure)
        except Exception:
            placeholder = "Objective figures unavailable in this notebook frontend."
            self.objective_card.children = (
                *self.objective_card.children,
                widgets.HTML(f"<p>{placeholder}</p>"),
            )
            self._line(placeholder)
        self.status = "objective_completed"
        self._line(f"Objective challenge complete: {run.termination_reason}")
        self._set_body()
        self._request_synthesis()

    def _request_synthesis(self) -> None:
        self.status = "synthesizing"
        try:
            result = self.controller.request_synthesis()
        except Exception as error:
            if self._known_failure(error) and self._synthesis_retryable():
                self._retry_card("Synthesis failed", error, "synthesis_failed", "Retry Synthesis", self._retry_synthesis)
            else:
                self._stop()
            return
        self.workflow_result = result
        output = widgets.Output()
        self.active_card = widgets.VBox((widgets.HTML("<h3>Evidence-Backed Conclusion</h3>"), output))
        self.status = "completed"
        self.retry_button = None
        self._line("Final synthesis complete")
        self._set_body()
        try:
            with output:
                demo_agent._display_conclusion(result)
        except Exception:
            placeholder = "Conclusion rendering unavailable in this notebook frontend."
            self.active_card.children = (*self.active_card.children, widgets.HTML(f"<p>{placeholder}</p>"))
            self._line(placeholder)

    def _retry_synthesis(self, button: widgets.Button) -> None:
        if self._busy or self.status != "synthesis_failed" or button is not self.retry_button or button.disabled:
            return
        if not self._synthesis_retryable():
            button.disabled = True
            self._stop()
            return
        self._busy = True
        button.disabled = True
        self.retry_button = None
        try:
            self._request_synthesis()
        finally:
            self._busy = False


def launch_interactive_workflow(
    user_goal: str,
    api_key: str,
    *,
    skill: str | None = None,
    client: Any = None,
    executors: dict[str, Any] | None = None,
) -> InteractiveWorkflow:
    controller = demo_agent.BoundedWorkflowController.create(
        user_goal,
        api_key,
        skill=skill,
        client=client,
        executors=executors,
        objective_required=True,
    )
    workflow = InteractiveWorkflow(controller)
    workflow.display()
    return workflow
