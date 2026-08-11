"""Guarded ipywidgets presentation for the bounded scientific controller."""

from __future__ import annotations

from dataclasses import fields
from html import escape
from io import BytesIO
from typing import Any

import ipywidgets as widgets
from IPython.display import display as ipython_display
from pydantic import BaseModel, ValidationError

import demo_agent
from command_receipts import CommandReceipt, command_receipt
from objective_challenge import (
    MAX_ATTEMPTS, ObjectiveActionMenu, ObjectiveAttempt, ObjectiveSwap,
    TerminationReason, accepted_maxima, build_action_menu, measure_panel,
    build_objective_evidence, objective_figures, score_key, target_is_achieved,
)


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
        self.objective_attempt_cards: tuple[widgets.HTML, ...] = ()
        self.objective_decisions: tuple[
            tuple[ObjectiveActionMenu, demo_agent.ObjectiveSelection, ObjectiveAttempt | None],
            ...,
        ] = ()
        self.objective_attempt_box = widgets.VBox()
        self.objective_output = widgets.VBox()
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

    def _stop_objective(
        self, error: demo_agent.ObjectiveCorrectionLimitError
    ) -> None:
        """Render a truthful terminal state for a known objective-agent failure."""
        if not isinstance(error, demo_agent.ObjectiveCorrectionLimitError):
            raise TypeError("Objective stops require a correction-limit error.")
        attempts = tuple(self.controller.objective_attempts)
        if (
            self.objective_card is None
            or len(self.objective_attempt_cards) != len(attempts)
        ):
            raise ValueError("Objective stop state does not match accepted attempts.")
        self._detach_observers()
        self.status = "objective_stopped"
        if self.retry_button is not None:
            self.retry_button.disabled = True
        if self.approve_button is not None:
            self.approve_button.disabled = True
        if self.objective_button is not None:
            self.objective_button.disabled = True
        for control in self.controls.values():
            control.disabled = True
        self.retry_button = None
        self.approve_button = None
        message = _safe_message(error)
        count_text = f"Accepted scientific attempts: {len(attempts)}."
        execution_text = "No additional scientific attempt was executed."
        self._line(
            f"Objective challenge stopped: {message} {count_text} {execution_text}"
        )
        self.objective_card.children = (
            *self.objective_card.children,
            widgets.HTML(
                "<b>Objective challenge stopped</b>"
                f"<p>{message}</p><p>{escape(count_text)} "
                f"{escape(execution_text)}</p>"
            ),
        )
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
                and self.controller.pending_objective_selection is None
                and self.controller.pending_action_menu is None
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
                and self.controller.objective_transport_retry_pending is True
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
                and self.controller.pending_objective_selection is None
                and type(self.controller.pending_action_menu) is ObjectiveActionMenu
                and bool(self.controller.pending_action_menu.actions)
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

    def _has_safe_terminal_objective(self) -> bool:
        """Accept only a complete controller-authored terminal run and O01 receipt."""
        try:
            run = self.controller.objective_run
            evidence = self.controller.objective_evidence
            return (
                run is not None
                and evidence is not None
                and evidence == build_objective_evidence(run)
                and self.controller.objective_context is run.context
                and self.controller.pending_action_menu is None
                and self.controller.pending_objective_selection is None
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
        figure_box = widgets.VBox()
        completion = widgets.HTML(
            f"<h3>Completed: {escape(result.stage)}</h3>{comparison}"
            f"<b>Approved tool call</b><pre>{escape(receipt.approved_tool_call)}</pre>"
            f"<b>{escape(receipt.scientific_label)}</b><pre>{escape(receipt.scientific_invocation)}</pre>"
            f"<p><b>Result metrics:</b> {escape(str(metrics))}</p>"
        )
        card = self.active_card
        if card is None:
            raise RuntimeError("Active proposal card was missing.")
        card.children = (*card.children, completion, figure_box)
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
        rendered_figures = []
        for figure in result.figures:
            try:
                rendered_figures.append(self._image_widget(figure))
            except Exception:
                placeholder = "Figure unavailable in this notebook frontend."
                rendered_figures.append(widgets.HTML(f"<p>{placeholder}</p>"))
                self._line(placeholder)
        figure_box.children = tuple(rendered_figures)

    @staticmethod
    def _image_widget(figure: Any) -> widgets.Image:
        """Persist a matplotlib or PIL/RDKit figure as standalone PNG bytes."""
        png = BytesIO()
        if callable(getattr(figure, "savefig", None)):
            figure.savefig(png, format="png", dpi=120, bbox_inches="tight")
        elif callable(getattr(figure, "save", None)):
            figure.save(png, format="PNG")
        else:
            raise TypeError("Unsupported persistent figure type.")
        value = png.getvalue()
        if not value.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Figure did not encode as PNG.")
        return widgets.Image(value=value, format="png")

    @staticmethod
    def _objective_summary_html(context, decisions, run=None) -> str:
        baseline = measure_panel(context, context.baseline_ids)
        for menu, selection, attempt in decisions:
            InteractiveWorkflow._validate_objective_decision(
                context, menu, selection, attempt
            )
        attempts = tuple(
            attempt for _menu, _selection, attempt in decisions if attempt is not None
        )
        display_values = tuple(
            InteractiveWorkflow._objective_display_values(
                attempt.score, context.target_score, attempt.selected_swap.score_delta
            )
            for attempt in attempts
        )
        if baseline.score_key != score_key(context.target_score):
            display_values = (
                InteractiveWorkflow._objective_display_values(
                    baseline.score,
                    context.target_score,
                    context.target_score - baseline.score,
                ),
                *display_values,
            )
        precision = max((values[3] for values in display_values), default=3)
        scientific = any(values[4] for values in display_values)
        baseline_pairs = " · ".join(
            f"{escape(first)} / {escape(second)}"
            for first, second in baseline.limiting_pairs
        )
        baseline_text, target_text, baseline_status = (
            InteractiveWorkflow._score_comparison(
                baseline.score, context.target_score
            )
        )
        baseline_row = (
            "<section style='border-left:3px solid #6c757d;padding:6px 10px;"
            "margin:6px 0' aria-label='Objective baseline'>"
            "<b>Step 0 · Measured baseline</b><br>"
            f"<small><b>Observe:</b> supplied panel {escape(str(baseline.selected_ids))}</small><br>"
            f"<small><b>Measure:</b> D_min {baseline_text} vs target {target_text} "
            f"· score_key={baseline.score_key} · co-limiting pairs {baseline_pairs}</small><br>"
            f"<small><b>Target status:</b> {escape(baseline_status)}</small>"
            "</section>"
        )
        current_score = attempts[-1].score if attempts else baseline.score
        current_progress = InteractiveWorkflow._objective_current_progress_html(
            context, current_score
        )
        terminal_actions = ""
        final = ""
        if run is not None:
            reason = TerminationReason(run.termination_reason)
            labels = {
                TerminationReason.TARGET_ACHIEVED: "Goal achieved; target achieved.",
                TerminationReason.BASELINE_ALREADY_OPTIMAL: (
                    "Baseline already optimal within the bounded pool."
                ),
                TerminationReason.ATTEMPT_LIMIT_REACHED: (
                    "Attempt limit reached before the target was achieved."
                ),
                TerminationReason.NO_LEGAL_IMPROVING_SWAP: (
                    "No legal improving swap remains in the deterministic action menu."
                ),
                TerminationReason.OBJECTIVE_CORRECTION_LIMIT: (
                    "Objective correction limit reached; no further selection was accepted."
                ),
                TerminationReason.OBJECTIVE_PROVIDER_FAILURE: (
                    "Objective provider failure; no further evaluation was performed."
                ),
                TerminationReason.EVALUATION_NOT_COMPLETED: (
                    "Evaluation not completed; the validated selection remains unmeasured."
                ),
            }
            label = labels[reason]
            if reason is TerminationReason.NO_LEGAL_IMPROVING_SWAP:
                source = baseline if not attempts else attempts[-1].measurement
                empty_menu = build_action_menu(
                    context, source, len(attempts)
                )
                if empty_menu.actions:
                    raise ValueError(
                        "No-legal terminal state rebuilt a nonempty action menu."
                    )
                terminal_actions = (
                    "<div aria-label='Terminal candidate actions'>"
                    + InteractiveWorkflow._objective_action_menu_html(
                        context, empty_menu
                    )
                    + "</div>"
                )
            final = f"<p><b>Outcome:</b> {escape(label)}</p>"
        return (
            InteractiveWorkflow._objective_story_style()
            + "<div aria-label='Objective decision ladder'>"
            "<p><b>Objective:</b> Select four MMFF94-parameter-eligible compounds "
            "that maximize minimum pairwise Morgan/Tanimoto distance.</p>"
            "<p><b>Constraints:</b> four unique supplied IDs · four distinct fused "
            "Butina clusters · fixed fingerprint evidence · at most three attempts</p>"
            "<p><b>Target:</b> D_min ≥ "
            f"{InteractiveWorkflow._objective_scalar(context.target_score, precision, scientific)} "
            "(80% of attainable improvement over baseline)</p>"
            f"{current_progress}{baseline_row}{terminal_actions}{final}</div>"
        )

    @staticmethod
    def _validate_objective_decision(context, menu, selection, attempt) -> None:
        if type(menu) is not ObjectiveActionMenu or menu != build_action_menu(
            context, menu.source, menu.accepted_attempt_count
        ):
            raise ValueError(
                "Displayed objective menu does not match deterministic controller state."
            )
        InteractiveWorkflow._objective_attempt_row(
            menu, selection, attempt, context
        )
        if attempt is None:
            return
        measured = measure_panel(context, attempt.selected_ids)
        if (
            attempt.measurement != measured
            or attempt.achieved
            != target_is_achieved(measured.score, context.target_score)
            or attempt.selected_swap.target_status
            != ("meets_target" if measured.achieved else "below_target")
        ):
            raise ValueError(
                "Displayed objective attempt does not match independent measurement."
            )

    @staticmethod
    def _score_comparison(first: float, second: float) -> tuple[str, str, str]:
        """Format a comparison with the same 1e-12 keys as the decision policy."""
        first_key, second_key = score_key(first), score_key(second)
        if first_key == second_key:
            tied = f"{first_key / 10**12:.12f}"
            return tied, tied, "tied at 1e-12 decision precision"
        for precision in range(3, 16):
            left, right = f"{first:.{precision}f}", f"{second:.{precision}f}"
            if left != right and ((float(left) > float(right)) == (first_key > second_key)):
                status = (
                    "above at 1e-12 decision precision"
                    if first_key > second_key
                    else "below at 1e-12 decision precision"
                )
                return left, right, status
        left, right = f"{first:.17e}", f"{second:.17e}"
        status = (
            "above at 1e-12 decision precision"
            if first_key > second_key
            else "below at 1e-12 decision precision"
        )
        return left, right, status

    @staticmethod
    def _objective_target_status(score: float, target: float) -> str:
        left, right, status = InteractiveWorkflow._score_comparison(score, target)
        return f"{left} vs target {right}: {status}"

    @staticmethod
    def _objective_action_menu_html(context, menu: ObjectiveActionMenu) -> str:
        if type(menu) is not ObjectiveActionMenu or menu != build_action_menu(
            context, menu.source, menu.accepted_attempt_count
        ):
            raise ValueError("Candidate actions require the exact deterministic menu.")
        if not menu.actions:
            return "<p><b>Candidate actions:</b> No legal improving candidate actions.</p>"
        rows = []
        for action in menu.actions:
            pairs = " · ".join(
                f"{escape(first)} / {escape(second)}"
                for first, second in action.limiting_pairs
            )
            resulting = ", ".join(escape(item) for item in action.resulting_ids)
            rows.append(
                "<section aria-label='Candidate action' style='border:1px solid #aaa;"
                "padding:6px 10px;margin:4px 0'>"
                f"<b>{escape(action.swap_id)}</b><br>"
                f"<small>State: {escape(menu.state_id)}</small><br>"
                f"<small>Resulting panel: {resulting}</small><br>"
                f"<small>Deterministic score: {action.predicted_score!r} "
                f"(score_key={action.predicted_score_key}) · Delta: {action.score_delta!r}</small><br>"
                f"<small>Resulting co-limiting pairs: {pairs}</small><br>"
                f"<small>Target status: {escape(InteractiveWorkflow._objective_target_status(action.predicted_score, context.target_score))}</small>"
                "</section>"
            )
        return "<div><b>Candidate actions</b>" + "".join(rows) + "</div>"

    @staticmethod
    def _objective_molecule_svg(context, molecules, molecule_id: str) -> str:
        """Draw one retained objective molecule as an inline RDKit SVG."""
        if molecules is None:
            return ""
        if type(molecules) not in {list, tuple}:
            raise ValueError("Objective molecule rendering requires retained molecules.")
        matches = tuple(
            candidate
            for candidate in context.candidates
            if candidate.molecule_id == molecule_id
        )
        if len(matches) != 1:
            raise ValueError("Objective molecule ID does not resolve uniquely.")
        candidate = matches[0]
        if not 0 <= candidate.molecule_index < len(molecules):
            raise ValueError("Objective molecule provenance is outside retained state.")
        molecule = molecules[candidate.molecule_index]
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            if not isinstance(molecule, Chem.Mol) or molecule.GetNumAtoms() < 1:
                raise ValueError
            svg = str(
                Draw.MolsToGridImage(
                    [molecule],
                    molsPerRow=1,
                    subImgSize=(220, 135),
                    useSVG=True,
                )
            )
        except Exception as error:
            raise ValueError("Objective molecule could not be drawn safely.") from error
        start = svg.find("<svg")
        end = svg.rfind("</svg>")
        if start < 0 or end < start:
            raise ValueError("RDKit did not return an inline SVG molecule drawing.")
        return svg[start : end + len("</svg>")]

    @staticmethod
    def _objective_molecule_tile(
        context,
        molecules,
        molecule_id: str,
        *,
        status: str = "",
        caption: str | None = None,
    ) -> str:
        svg = InteractiveWorkflow._objective_molecule_svg(
            context, molecules, molecule_id
        )
        drawing = (
            svg
            if svg
            else "<span class='odc-structure-unavailable'>Structure unavailable</span>"
        )
        safe_id = escape(molecule_id, quote=True)
        return (
            f"<div class='odc-molecule {escape(status, quote=True)}' "
            f"data-molecule-id='{safe_id}'>"
            f"<div class='odc-drawing'>{drawing}</div>"
            f"<span>{escape(caption or molecule_id)}</span></div>"
        )

    @staticmethod
    def _objective_progress_html(context, score: float, attempt_number: int) -> str:
        required = context.target_score - context.baseline_score
        achieved = score - context.baseline_score
        fraction = 1.0 if score_key(required) == 0 else achieved / required
        percent = max(0.0, min(100.0, 100.0 * fraction))
        remaining = max(0.0, context.target_score - score)
        score_text, target_text, remaining_text, _precision, _scientific = (
            InteractiveWorkflow._objective_display_values(
                score, context.target_score, remaining
            )
        )
        status = InteractiveWorkflow._score_comparison(score, context.target_score)[2]
        percent_text = f"{percent:.0f}%"
        if score_key(score) < score_key(context.target_score) and percent_text == "100%":
            percent_text = "<100%"
        baseline_text = InteractiveWorkflow._score_comparison(
            context.baseline_score, context.target_score
        )[0]
        return (
            f"<section class='odc-progress' data-objective-progress='{attempt_number}' "
            f"aria-label='Attempt {attempt_number} progress'>"
            "<div class='odc-progress-head'>"
            f"<b>After Attempt {attempt_number}: D_min = {score_text}</b>"
            f"<span>{remaining_text.lstrip('+')} from target · {escape(percent_text)} of required improvement achieved</span>"
            "</div><div class='odc-progress-track'>"
            f"<span class='odc-progress-fill' style='width:{percent:.1f}%'></span>"
            "</div><div class='odc-progress-labels'>"
            f"<span class='odc-progress-baseline'><b>Baseline</b> {baseline_text}</span>"
            f"<span class='odc-progress-current'><b>Current</b> {score_text}</span>"
            f"<span class='odc-progress-target'><b>Target</b> {target_text}</span>"
            f"</div><small>{escape(status)}</small></section>"
        )

    @staticmethod
    def _objective_current_progress_html(context, score: float) -> str:
        required = context.target_score - context.baseline_score
        achieved = score - context.baseline_score
        fraction = 1.0 if score_key(required) == 0 else achieved / required
        percent = max(0.0, min(100.0, 100.0 * fraction))
        remaining = max(0.0, context.target_score - score)
        score_text, target_text, remaining_text, _precision, _scientific = (
            InteractiveWorkflow._objective_display_values(
                score, context.target_score, remaining
            )
        )
        status = InteractiveWorkflow._score_comparison(score, context.target_score)[2]
        percent_text = f"{percent:.0f}%"
        if score_key(score) < score_key(context.target_score) and percent_text == "100%":
            percent_text = "<100%"
        baseline_text = InteractiveWorkflow._score_comparison(
            context.baseline_score, context.target_score
        )[0]
        return (
            "<section class='odc-progress odc-current-progress' "
            "data-objective-current-progress='true' aria-label='Current objective progress'>"
            "<div class='odc-progress-head'>"
            f"<b>Current D_min = {score_text}</b>"
            f"<span>{remaining_text.lstrip('+')} from target · {escape(percent_text)} of required improvement achieved</span>"
            "</div><div class='odc-progress-track'>"
            f"<span class='odc-progress-fill' style='width:{percent:.1f}%'></span>"
            "</div><div class='odc-progress-labels'>"
            f"<span class='odc-progress-baseline'><b>Baseline</b> {baseline_text}</span>"
            f"<span class='odc-progress-current'><b>Current</b> {score_text}</span>"
            f"<span class='odc-progress-target'><b>Target</b> {target_text}</span>"
            f"</div><small>{escape(status)}</small></section>"
        )

    @staticmethod
    def _objective_unmeasured_progress_html(
        context, score: float, attempt_number: int
    ) -> str:
        required = context.target_score - context.baseline_score
        achieved = score - context.baseline_score
        fraction = 1.0 if score_key(required) == 0 else achieved / required
        percent = max(0.0, min(100.0, 100.0 * fraction))
        remaining = max(0.0, context.target_score - score)
        score_text, target_text, remaining_text, _precision, _scientific = (
            InteractiveWorkflow._objective_display_values(
                score, context.target_score, remaining
            )
        )
        baseline_text = InteractiveWorkflow._score_comparison(
            context.baseline_score, context.target_score
        )[0]
        return (
            f"<section class='odc-progress' data-objective-progress='{attempt_number}' "
            "data-progress-status='unmeasured' aria-label='Last measured objective progress'>"
            "<div class='odc-progress-head'>"
            f"<b>Last measured D_min = {score_text}</b>"
            "<span>no progress update · selection was not evaluated</span>"
            "</div><div class='odc-progress-track'>"
            f"<span class='odc-progress-fill' style='width:{percent:.1f}%'></span>"
            "</div><div class='odc-progress-labels'>"
            f"<span class='odc-progress-baseline'><b>Baseline</b> {baseline_text}</span>"
            f"<span class='odc-progress-current'><b>Current</b> {score_text}</span>"
            f"<span class='odc-progress-target'><b>Target</b> {target_text}</span>"
            f"</div><small>{remaining_text.lstrip('+')} remained before the target</small></section>"
        )

    @staticmethod
    def _objective_story_style() -> str:
        return """
<style>
.odc-attempt{border-left:4px solid #6c757d;padding:12px;margin:10px 0 18px;background:#181818;color:#f2f2f2}
.odc-attempt.odc-revise{border-left-color:#D68A00}.odc-attempt.odc-achieved{border-left-color:#76B900}
.odc-attempt h4{margin:0 0 8px}.odc-steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid #4a4a4a}
.odc-step{min-width:0;padding:10px;border-right:1px solid #4a4a4a}.odc-step:last-child{border-right:0}.odc-step-title{display:flex;gap:7px;align-items:center;margin-bottom:8px;font-weight:600}.odc-step-number{display:inline-grid;place-items:center;width:22px;height:22px;background:#76B900;color:#101010}
.odc-molecules{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.odc-molecule{min-width:0;padding:5px;background:linear-gradient(145deg,#fff,#edf1e9);border:2px solid #d2d7cf;border-radius:8px;box-shadow:0 2px 7px rgba(0,0,0,.24);text-align:center;color:#202020}.odc-molecule.odc-weak,.odc-molecule.odc-out{border-color:#E57373}.odc-molecule.odc-in{border-color:#76B900}.odc-molecule.odc-limit{border-color:#F2B84B}.odc-drawing{height:72px;display:grid;place-items:center;overflow:hidden}.odc-drawing svg{width:100%;height:100%;display:block}.odc-molecule span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:600}
.odc-candidates{display:grid;gap:6px}.odc-candidate{display:grid;grid-template-columns:62px 1fr;gap:7px;align-items:center;padding:5px;background:#f4f5f1;color:#202020;border:1px solid #d2d7cf;border-radius:7px}.odc-candidate.odc-selected{border:2px solid #76B900}.odc-candidate .odc-molecule{padding:2px;border:0;box-shadow:none}.odc-candidate .odc-drawing{height:42px}.odc-candidate small{display:block}.odc-choice{display:grid;grid-template-columns:1fr 22px 1fr;gap:4px;align-items:center}.odc-arrow{text-align:center;color:#9bd43e;font-size:22px}.odc-step p{margin:8px 0 0;color:#c7c7c7;font-size:12px}.odc-score{margin-top:8px;color:#9bd43e;font-size:18px;font-weight:600}
.odc-explain{display:grid;grid-template-columns:1.2fr 1fr;gap:12px;margin-top:12px}.odc-explain-panel{padding:12px;background:#222;border:1px solid #4a4a4a}.odc-change{display:grid;grid-template-columns:1fr 34px 1fr;gap:8px;align-items:center}.odc-change .odc-drawing{height:105px}.odc-why{margin:0;padding-left:20px}.odc-why li{margin:8px 0}
.odc-progress{margin-top:11px;padding:11px 12px;background:#222;border:1px solid #4a4a4a;color:#f2f2f2}.odc-progress-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.odc-progress-head b{color:#f2f2f2!important}.odc-progress-head span{color:#9bd43e}.odc-progress-track{position:relative;height:18px;margin-top:6px}.odc-progress-track:before{content:'';position:absolute;left:0;right:0;top:7px;height:6px;background:#383838}.odc-progress-fill{position:absolute;left:0;top:7px;height:6px;background:#76B900}.odc-progress-labels{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:3px;font-size:12px}.odc-progress-labels span{position:static;transform:none;white-space:nowrap}.odc-progress-baseline{text-align:left;color:#bbb}.odc-progress-current{text-align:center;color:#9bd43e}.odc-progress-target{text-align:right;color:#F2B84B}.odc-progress-labels b{color:inherit!important}.odc-progress small{display:block;margin-top:5px;color:#c7c7c7!important}
@media(max-width:900px){.odc-steps{grid-template-columns:1fr}.odc-step{border-right:0;border-bottom:1px solid #4a4a4a}.odc-step:last-child{border-bottom:0}.odc-explain{grid-template-columns:1fr}.odc-progress-head{display:block}.odc-progress-head span{display:block;margin-top:4px}}
</style>"""

    @staticmethod
    def _objective_attempt_row(
        menu,
        selection,
        attempt=None,
        context=None,
        *,
        molecules=None,
        show_explanation=False,
    ) -> str:
        """Render one validated decision as a concise five-step molecule story."""
        if type(menu) is not ObjectiveActionMenu or type(selection) is not demo_agent.ObjectiveSelection:
            raise ValueError("Objective rows require exact public menu and selection types.")
        source_ids = menu.source.selected_ids

        def valid_panel(panel) -> bool:
            return (
                type(panel) is tuple
                and len(panel) == 4
                and len(set(panel)) == 4
                and all(type(item) is str and item for item in panel)
            )

        def valid_pairs(pairs, panel) -> bool:
            return (
                type(pairs) is tuple
                and bool(pairs)
                and all(
                    type(pair) is tuple
                    and len(pair) == 2
                    and pair[0] != pair[1]
                    and all(type(item) is str and item in panel for item in pair)
                    for pair in pairs
                )
            )

        if (
            not valid_panel(source_ids)
            or menu.source.score_key != score_key(menu.source.score)
            or not valid_pairs(menu.source.limiting_pairs, source_ids)
            or any(type(action) is not ObjectiveSwap for action in menu.actions)
        ):
            raise ValueError("Objective menu source is not canonical measured evidence.")
        observed = tuple(tuple(pair) for pair in selection.observed_limiting_pairs)
        selected = next(
            (action for action in menu.actions if action.swap_id == selection.swap_id), None
        )
        if (
            selection.state_id != menu.state_id
            or observed != menu.source.limiting_pairs
            or selection.decision_rule != "maximize_predicted_minimum_distance"
            or selected is None
            or selected not in accepted_maxima(menu)
        ):
            raise ValueError("Objective selection does not match its displayed menu.")
        if (
            not valid_panel(selected.resulting_ids)
            or selected.swap_id
            != f"{selected.replace_id}->{selected.replacement_id}"
            or selected.replace_id not in source_ids
            or selected.replacement_id in source_ids
            or set(selected.resulting_ids)
            != (set(source_ids) - {selected.replace_id}) | {selected.replacement_id}
            or selected.predicted_score_key != score_key(selected.predicted_score)
            or selected.predicted_score_key <= menu.source.score_key
            or selected.score_delta != selected.predicted_score - menu.source.score
            or selected.score_delta <= 0.0
            or not valid_pairs(selected.limiting_pairs, selected.resulting_ids)
            or selected.limiting_pair != selected.limiting_pairs[0]
            or selected.target_status not in {"below_target", "meets_target"}
            or not all(
                selected.replace_id in pair for pair in menu.source.limiting_pairs
            )
        ):
            raise ValueError("Objective selected action is not canonical policy evidence.")
        objective_context = context if type(context) is demo_agent.ObjectiveContext else None
        target_score = (
            objective_context.target_score if objective_context is not None else None
        )
        source_limiters = {
            molecule_id for pair in menu.source.limiting_pairs for molecule_id in pair
        }
        source_tiles = "".join(
            InteractiveWorkflow._objective_molecule_tile(
                objective_context,
                molecules,
                molecule_id,
                status="odc-weak" if molecule_id in source_limiters else "",
            )
            for molecule_id in source_ids
        )
        candidate_rows = []
        for action in menu.actions[:4]:
            tile = InteractiveWorkflow._objective_molecule_tile(
                objective_context,
                molecules,
                action.replacement_id,
                status="odc-in" if action.swap_id == selected.swap_id else "",
            )
            candidate_rows.append(
                "<div class='odc-candidate "
                f"{'odc-selected' if action.swap_id == selected.swap_id else ''}' "
                "aria-label='Candidate molecule action'>"
                f"{tile}<div><b>{escape(action.swap_id)}</b>"
                f"<small>D_min {action.predicted_score!r} · Δ {action.score_delta:+.3f}</small>"
                "</div></div>"
            )
        omitted_count = max(0, len(menu.actions) - len(candidate_rows))
        omitted = (
            f"<p>+{omitted_count} additional legal action{'s' if omitted_count != 1 else ''}</p>"
            if omitted_count else ""
        )
        maximum_key = max(action.predicted_score_key for action in menu.actions)
        maximum_count = sum(
            action.predicted_score_key == maximum_key for action in menu.actions
        )
        maximum_description = (
            f"one of {maximum_count} tied-max actions at 1e-12 decision precision"
            if maximum_count > 1
            else "the unique argmax at 1e-12 decision precision"
        )
        attempt_number = menu.accepted_attempt_count + 1
        if attempt is None:
            accent = "#6c757d"
            outcome = "Evaluation not completed"
            executed_ids = source_ids
            execute_detail = (
                "Not executed. The selection was validated, but the source panel remains active."
            )
            measured_ids = source_ids
            measured_limiters = source_limiters
            measured_score = menu.source.score
            measured_pairs = "Measurement unavailable"
            measure_detail = (
                "Evaluation was not completed. The selection is validated but unmeasured; "
                "the last measured source panel is shown, and no new D_min or target "
                "result is claimed."
            )
        else:
            if (
                type(attempt) is not ObjectiveAttempt
                or attempt.attempt_number != attempt_number
                or attempt.state_id != menu.state_id
                or attempt.selected_swap != selected
                or attempt.selected_ids != selected.resulting_ids
                or attempt.score != selected.predicted_score
                or attempt.score_key != selected.predicted_score_key
                or score_key(attempt.score) != attempt.score_key
                or attempt.limiting_pair != selected.limiting_pair
                or attempt.limiting_pairs != selected.limiting_pairs
                or attempt.constraints_passed is not True
                or attempt.achieved != (selected.target_status == "meets_target")
            ):
                raise ValueError("Committed attempt does not match its menu-bound selection.")
            accent = "#76B900" if attempt.achieved else "#D68A00"
            outcome = "Goal achieved" if attempt.achieved else "Revise"
            executed_ids = selected.resulting_ids
            execute_detail = "Python validates four unique IDs and four clusters."
            measured_ids = attempt.selected_ids
            measured_limiters = {
                molecule_id for pair in attempt.limiting_pairs for molecule_id in pair
            }
            measured_pairs = " · ".join(
                f"{escape(first)} / {escape(second)}" for first, second in attempt.limiting_pairs
            )
            if objective_context is None:
                similarities = "context required for raw per-pair similarities"
            else:
                positions = {
                    candidate.molecule_id: index
                    for index, candidate in enumerate(objective_context.candidates)
                }
                similarities = " · ".join(
                    f"{escape(first)} / {escape(second)}: "
                    f"{1.0 - float(objective_context.distance_matrix[positions[first], positions[second]])!r}"
                    for first, second in attempt.limiting_pairs
                )
            target_comparison = (
                InteractiveWorkflow._objective_target_status(
                    attempt.score, target_score
                )
                if type(target_score) is float
                else selected.target_status
            )
            measured_score = attempt.score
            measure_detail = (
                f"Co-limiting pair{'s' if len(attempt.limiting_pairs) != 1 else ''}: "
                f"{measured_pairs}. Limiting Tanimoto similarities: {similarities}. "
                f"Constraints passed. {target_comparison}."
            )
        chosen_out = InteractiveWorkflow._objective_molecule_tile(
            objective_context, molecules, selected.replace_id, status="odc-out",
            caption=f"{selected.replace_id} out",
        )
        chosen_in = InteractiveWorkflow._objective_molecule_tile(
            objective_context, molecules, selected.replacement_id, status="odc-in",
            caption=f"{selected.replacement_id} in",
        )
        executed_tiles = "".join(
            InteractiveWorkflow._objective_molecule_tile(
                objective_context,
                molecules,
                molecule_id,
                status="odc-in" if molecule_id == selected.replacement_id else "",
            )
            for molecule_id in executed_ids
        )
        measured_tiles = "".join(
            InteractiveWorkflow._objective_molecule_tile(
                objective_context,
                molecules,
                molecule_id,
                status="odc-limit" if molecule_id in measured_limiters else "",
            )
            for molecule_id in measured_ids
        )
        explain = ""
        if show_explanation:
            explain = (
                "<div class='odc-explain'>"
                "<section class='odc-explain-panel' aria-label='Molecular change'>"
                "<h4>Molecular change</h4><div class='odc-change'>"
                f"{chosen_out}<span class='odc-arrow'>→</span>{chosen_in}</div></section>"
                "<section class='odc-explain-panel' aria-label='Why this choice'>"
                "<h4>Why this choice?</h4><ol class='odc-why'>"
                f"<li>It was {escape(maximum_description)}.</li>"
                "<li>It removes one molecule from every current co-limiting pair.</li>"
                "<li>It preserves four unique molecules from four fused Butina clusters.</li>"
                "</ol></section></div>"
            )
        progress = (
            InteractiveWorkflow._objective_progress_html(
                objective_context, measured_score, attempt_number
            )
            if attempt is not None
            else InteractiveWorkflow._objective_unmeasured_progress_html(
                objective_context, measured_score, attempt_number
            )
        )
        css_class = (
            "odc-achieved" if attempt is not None and attempt.achieved
            else "odc-revise" if attempt is not None else "odc-unmeasured"
        )
        return (
            InteractiveWorkflow._objective_story_style()
            + f"<article class='odc-attempt {css_class}' style='border-left-color:{accent}' "
            "aria-label='Objective attempt'>"
            f"<h4>Attempt {attempt_number} · {escape(outcome)}</h4>"
            "<div class='odc-steps'>"
            "<section class='odc-step' aria-label='Objective step'><div class='odc-step-title'><span class='odc-step-number'>1</span><span>Observe panel</span></div>"
            f"<div class='odc-molecules'>{source_tiles}</div><p>D_min {menu.source.score!r} · limiting pair{'s' if len(menu.source.limiting_pairs) != 1 else ''}: {escape(str(menu.source.limiting_pairs))}</p></section>"
            "<section class='odc-step' aria-label='Objective step'><div class='odc-step-title'><span class='odc-step-number'>2</span><span>Candidate menu</span></div>"
            f"<div class='odc-candidates'>{''.join(candidate_rows)}</div>{omitted}</section>"
            "<section class='odc-step' aria-label='Objective step'><div class='odc-step-title'><span class='odc-step-number'>3</span><span>Agent chooses</span></div>"
            f"<div class='odc-choice'>{chosen_out}<span class='odc-arrow'>→</span>{chosen_in}</div><p>Highest calculated D_min in the validated menu.</p></section>"
            "<section class='odc-step' aria-label='Objective step'><div class='odc-step-title'><span class='odc-step-number'>4</span><span>Execute panel</span></div>"
            f"<div class='odc-molecules'>{executed_tiles}</div><p>{escape(execute_detail)}</p></section>"
            "<section class='odc-step' aria-label='Objective step'><div class='odc-step-title'><span class='odc-step-number'>5</span><span>Measure panel</span></div>"
            f"<div class='odc-molecules'>{measured_tiles}</div><div class='odc-score'>D_min {measured_score!r}</div><p>{escape(measure_detail)}</p></section>"
            f"</div>{explain}{progress}</article>"
        )

    @staticmethod
    def _objective_precision(score: float, target: float, delta: float) -> int:
        """Use the shortest honest fixed precision for one decision measurement."""
        for precision in range(3, 16):
            score_text = f"{score:.{precision}f}"
            target_text = f"{target:.{precision}f}"
            delta_text = f"{delta:+.{precision}f}"
            indistinguishable_scores = score != target and score_text == target_text
            erased_delta = delta != 0.0 and float(delta_text) == 0.0
            if not indistinguishable_scores and not erased_delta:
                return precision
        return 15

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
                self.objective_decisions,
                self.controller.objective_run,
            )
        )
        button = widgets.Button(
            description="Run Objective Challenge", button_style="success"
        )
        button.on_click(self._run_objective_challenge)
        self.objective_button = button
        self.objective_attempt_cards = ()
        self.objective_decisions = ()
        self.objective_attempt_box = widgets.VBox()
        self.objective_output = widgets.VBox()
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

    def _render_objective_decisions(self) -> None:
        context = self.controller.objective_context
        if context is None:
            raise ValueError("Objective decision rendering requires controller context.")
        summary_html = self._objective_summary_html(
            context, self.objective_decisions, self.controller.objective_run
        )
        molecules = getattr(self.controller.session.state, "molecules", None)
        cards = []
        last_index = len(self.objective_decisions) - 1
        for index, (menu, selection, attempt) in enumerate(self.objective_decisions):
            row = self._objective_attempt_row(
                menu,
                selection,
                attempt,
                self.controller.objective_context,
                molecules=molecules,
                show_explanation=index == last_index,
            )
            cards.append(widgets.HTML(row))
        self.objective_attempt_cards = tuple(cards)
        self.objective_attempt_box.children = self.objective_attempt_cards
        self.objective_summary.value = summary_html
        self._set_body()

    def _append_objective_attempt(self, menu, selection, attempt) -> None:
        if (
            type(menu) is not ObjectiveActionMenu
            or type(selection) is not demo_agent.ObjectiveSelection
            or type(attempt) is not ObjectiveAttempt
            or not self.controller.objective_attempts
            or self.controller.objective_attempts[-1] != attempt
        ):
            raise ValueError("Objective UI requires the exact committed controller attempt.")
        self.objective_decisions = (
            *self.objective_decisions,
            (menu, selection, attempt),
        )
        try:
            self._render_objective_decisions()
        except Exception:
            # Rebuild only the presentation from the retained controller-bound ledger.
            self._render_objective_decisions()
        result_label = "Goal achieved" if attempt.achieved else "Revise"
        self._line(
            f"Objective attempt {attempt.attempt_number}: score={attempt.score!r}; "
            f"score_key={attempt.score_key}; limiting_pairs={attempt.limiting_pairs}; "
            f"result={result_label}"
        )

    def _append_objective_evaluation_failure(self, menu, selection) -> None:
        self.objective_decisions = (
            *self.objective_decisions,
            (menu, selection, None),
        )
        self._render_objective_decisions()
        self._line("Objective evaluation not completed; validated selection was not measured.")

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
                menu = self.controller.pending_action_menu
                if type(menu) is not ObjectiveActionMenu:
                    raise ValueError("The controller has no exact pending action menu.")
                selection = self.controller.request_objective_attempt()
                attempt = self.controller.execute_objective_attempt(selection)
                self._append_objective_attempt(menu, selection, attempt)
            self._finish_objective_challenge()
        except demo_agent.ObjectiveEvaluationError:
            try:
                if (
                    type(menu) is not ObjectiveActionMenu
                    or type(selection) is not demo_agent.ObjectiveSelection
                    or self.controller.objective_run is None
                    or self.controller.objective_run.termination_reason
                    != TerminationReason.EVALUATION_NOT_COMPLETED
                    or self.controller.pending_action_menu is not None
                    or self.controller.pending_objective_selection is not None
                ):
                    raise ValueError("Objective evaluation failure state is incomplete.")
                self._append_objective_evaluation_failure(menu, selection)
                self._finish_objective_challenge()
            except Exception:
                self._stop()
        except Exception as error:
            if self._known_failure(error) and self._objective_retryable():
                self._retry_card(
                    "Objective proposal failed",
                    error,
                    "objective_failed",
                    "Retry Objective Proposal",
                    self._retry_objective,
                )
            elif self._has_safe_terminal_objective():
                try:
                    self._finish_objective_challenge()
                except Exception:
                    self._stop()
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
            self.objective_decisions,
            run,
        )
        try:
            figures = objective_figures(run, self.controller.session.state)
            self.objective_output.children = tuple(
                self._image_widget(figure) for figure in figures
            )
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
        try:
            self._render_workflow_result(result.conclusion)
        except Exception:
            self.active_card = widgets.VBox((widgets.HTML(
                "<h3>Evidence-Backed Conclusion</h3>"
                "<p>Conclusion rendering unavailable in this notebook frontend. "
                "The validated workflow result remains available.</p>"
            ),))
            self._line("Conclusion rendering unavailable in this notebook frontend.")
        self.status = "completed"
        self.retry_button = None
        self._line("Final synthesis complete")
        self._set_body()

    def _render_workflow_result(self, conclusion: Any) -> None:
        if isinstance(conclusion, demo_agent.EvidenceControlledConclusion):
            conclusion = demo_agent.validate_evidence_controlled_conclusion(conclusion)
            snapshot = conclusion.evidence_snapshot
            catalog = demo_agent.build_finding_catalog_from_snapshot(snapshot)
            summary = conclusion.measured_summary
            if conclusion.finding_selection_status == "finding_selection_unavailable":
                global_label = "agent-selected emphasis unavailable"
            else:
                global_label = "evidence-controlled measured findings"
            facts = "".join(f"<li>{escape(fact)}</li>" for fact in summary.facts)
            measured_summary = self._measured_summary_html(summary)
            findings = []
            for finding in conclusion.ordered_findings:
                demo_agent.validate_finding(finding, snapshot)
                if conclusion.finding_selection_status == "finding_selection_unavailable":
                    label = "deterministic fallback finding"
                elif len(catalog.ids_for_theme(finding.theme)) > 1:
                    label = "agent-selected evidence emphasis"
                else:
                    label = "required measured finding"
                findings.append(
                    f"<h4>{escape(finding.theme.replace('_', ' ').title())}</h4>"
                    f"<p><b>{escape(label)}</b></p>"
                    f"<p>{escape(finding.text)}</p>"
                )
            content = widgets.HTML(
                "<h3>Evidence-Backed Conclusion</h3>"
                "<h3>Measured evidence-controlled conclusion</h3>"
                f"<h4>{escape(summary.headline)}</h4>"
                f"<p><b>{escape(global_label)}</b></p>"
                f"<ul>{facts}</ul>{measured_summary}{''.join(findings)}"
            )
        else:
            sections = "".join(
                f"<h4>{escape(section.theme.replace('_', ' ').title())}</h4>"
                f"<p>{escape(demo_agent._presentation_text(section.prose))}</p>"
                for section in conclusion.sections
            )
            content = widgets.HTML(
                "<h3>Evidence-Backed Conclusion</h3>"
                "<h3>Schema-checked scientific conclusion</h3>"
                f"<h4>{escape(demo_agent._presentation_text(conclusion.headline))}</h4>"
                "<p>Python checks the response structure before rendering; Nemotron's "
                "qualitative interpretation is not automatically fact-verified.</p>"
                "<p>Python-rendered methods: 3D conformers use ETKDGv3; energies use MMFF94.</p>"
                f"{sections}"
            )
        self.active_card = widgets.VBox((content,))
        self._set_body()

    @staticmethod
    def _measured_summary_html(summary: demo_agent.MeasuredSummary) -> str:
        """Render the complete required quantitative conclusion state deterministically."""
        if type(summary) is not demo_agent.MeasuredSummary:
            raise ValueError("Measured summary rendering requires the exact summary type.")
        limiting_pairs = "; ".join(
            f"{first} / {second}: Tanimoto {similarity}"
            for (first, second), similarity in zip(
                summary.limiting_pairs,
                summary.limiting_similarities,
                strict=True,
            )
        )
        rows = []
        for field in fields(summary):
            raw_value = getattr(summary, field.name)
            value = str(raw_value)
            if field.name == "optimization_comparison_scope":
                value = (
                    "within molecule among converged sampled conformers; "
                    f"recorded scope: {raw_value}"
                )
            elif field.name == "limiting_pairs":
                value = f"{raw_value}; paired similarities: {limiting_pairs}"
            elif (
                field.name == "target_margin"
                and summary.final_distance != summary.target_distance
                and score_key(summary.final_distance) == score_key(summary.target_distance)
            ):
                value = f"{raw_value}; tied at 1e-12 decision precision"
            rows.append((field.name, field.name.replace("_", " ").title(), value))
        body = "".join(
            f'<tr data-field="{escape(field_name)}">'
            f"<th>{escape(label)}</th><td>{escape(value)}</td></tr>"
            for field_name, label, value in rows
        )
        return (
            '<h4>Deterministic Measured Summary</h4>'
            f'<table class="measured-summary"><tbody>{body}</tbody></table>'
        )

    def reconstruct_completed_view(self) -> widgets.VBox:
        """Rebuild the persistent conclusion from the retained result only."""
        if self.workflow_result is None:
            raise ValueError("No completed workflow result is available.")
        self._render_workflow_result(self.workflow_result.conclusion)
        assert self.active_card is not None
        return self.active_card

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
