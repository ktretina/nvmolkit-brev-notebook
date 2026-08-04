"""Guarded ipywidgets presentation for the bounded scientific controller."""

from __future__ import annotations

from html import escape
from typing import Any

import ipywidgets as widgets
from IPython.display import display as ipython_display
from pydantic import BaseModel, ValidationError

import demo_agent
from command_receipts import CommandReceipt, command_receipt


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
        self.status = "ready"
        self.controls: dict[str, widgets.Widget] = {}
        self.approve_button: widgets.Button | None = None
        self.retry_button: widgets.Button | None = None
        self.transcript_text = ""
        self.completed_cards: tuple[widgets.VBox, ...] = ()
        self.active_card: widgets.VBox | None = None
        self._active_proposal: demo_agent.StageProposal | None = None
        self._approved: BaseModel | None = None
        self._busy = False
        self.start_button = widgets.Button(description="Start Agent", button_style="primary")
        self.start_button.on_click(lambda _button: self.start())
        self._body = widgets.VBox()
        self.root = widgets.VBox((self.start_button, self._body))

    def display(self) -> widgets.VBox:
        ipython_display(self.root)
        return self.root

    def _line(self, text: str) -> None:
        self.transcript_text += text + "\n"

    def _set_body(self) -> None:
        children = list(self.completed_cards)
        if self.active_card is not None:
            children.append(self.active_card)
        self._body.children = tuple(children)

    def _error_card(self, title: str, error: Exception, retry_label: str | None, callback=None) -> None:
        self.status = "error" if retry_label else "stopped"
        message = _safe_message(error)
        self._line(f"{title}: {message}")
        children: list[widgets.Widget] = [widgets.HTML(f"<b>{escape(title)}</b><p>{message}</p>")]
        self.retry_button = None
        if retry_label and callback is not None:
            button = widgets.Button(description=retry_label)
            button.on_click(lambda _button: callback())
            children.append(button)
            self.retry_button = button
        self.active_card = widgets.VBox(tuple(children))
        self._set_body()

    def start(self) -> None:
        if self._busy or self.status != "ready":
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
            self.active_card = widgets.VBox((widgets.HTML(f"<h3>Fixed workflow plan</h3><ol>{lines}</ol>"),))
            self._set_body()
            self._request_proposal()
        except Exception as error:
            self._error_card("Plan request failed", error, "Retry Plan", self._retry_plan)
        finally:
            self._busy = False

    def _retry_plan(self) -> None:
        if self._busy:
            return
        self.status = "ready"
        self.retry_button.disabled = True
        self.start_button.disabled = False
        self.start()

    def _evidence_summary(self) -> str:
        results = getattr(self.controller, "stage_results", ())
        if not results:
            return "No prior scientific results; fixed input and plan only."
        return "; ".join(f"{result.stage}: {len(result.summary)} metrics" for result in results)

    def _request_proposal(self) -> None:
        self.status = "requesting_proposal"
        try:
            proposal = self.controller.request_next_stage()
            self._show_proposal(proposal)
        except Exception as error:
            self._error_card("Stage proposal failed", error, "Retry Proposal", self._request_proposal)

    def _show_proposal(self, proposal: demo_agent.StageProposal) -> None:
        self._active_proposal = proposal
        self._approved = None
        self.controls = controls_for(proposal)
        proposed_receipt = command_receipt(proposal.stage, proposal.arguments)
        preview = widgets.HTML()

        def update_preview(_change=None):
            try:
                approved = _approved_model(proposal, self.controls)
                preview.value = "<b>Approved-call preview</b><pre>" + escape(
                    command_receipt(proposal.stage, approved).approved_tool_call
                ) + "</pre>"
            except (ValidationError, ValueError):
                preview.value = "<b>Approved-call preview unavailable</b>"

        for control in self.controls.values():
            control.observe(update_preview, names="value")
        update_preview()
        button = widgets.Button(description="Approve & Run", button_style="success")
        button.on_click(lambda _button: self._approve(button))
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
        if self._busy or original_button.disabled or self.status != "awaiting_approval":
            return
        self._busy = True
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
                self._request_synthesis()
            else:
                self._request_proposal()
        except Exception as error:
            if self._execution_is_retryable(proposal):
                self._error_card("Scientific execution failed", error, "Retry Execution", self._retry_execution)
            else:
                self._error_card("Workflow stopped", error, None)
        finally:
            self._busy = False

    def _execution_is_retryable(self, proposal) -> bool:
        if proposal is None or self._approved is None:
            return False
        try:
            return self.controller.pending is proposal and self.controller.session.eligible_tool_name() == proposal.stage
        except Exception:
            return False

    def _retry_execution(self) -> None:
        if self._busy or self._approved is None or self._active_proposal is None:
            return
        self._busy = True
        self.retry_button.disabled = True
        proposal, approved = self._active_proposal, self._approved
        try:
            receipt = command_receipt(proposal.stage, approved)
            result = self.controller.execute_pending(approved)
            self._complete_card(proposal, approved, receipt, result)
            if len(self.completed_cards) == len(demo_agent.STAGES):
                self._request_synthesis()
            else:
                self._request_proposal()
        except Exception as error:
            if self._execution_is_retryable(proposal):
                self._error_card("Scientific execution failed", error, "Retry Execution", self._retry_execution)
            else:
                self._error_card("Workflow stopped", error, None)
        finally:
            self._busy = False

    def _complete_card(self, proposal, approved, receipt: CommandReceipt, result) -> None:
        proposed_values, approved_values = _parameters(proposal.arguments), _parameters(approved)
        changed = proposed_values != approved_values
        comparison = f"<p><b>Proposed:</b> {escape(str(proposed_values))}<br><b>Approved:</b> {escape(str(approved_values))}</p>" if changed else "<p>Proposal approved unchanged.</p>"
        metrics = {key: result.summary[key] for key in demo_agent._STAGE_METRICS[result.stage] if key in result.summary}
        output = widgets.Output()
        with output:
            for figure in result.figures:
                demo_agent._display_figure(figure)
        card = widgets.VBox((widgets.HTML(
            f"<h3>Completed: {escape(result.stage)}</h3>{comparison}"
            f"<b>Approved tool call</b><pre>{escape(receipt.approved_tool_call)}</pre>"
            f"<b>{escape(receipt.scientific_label)}</b><pre>{escape(receipt.scientific_invocation)}</pre>"
            f"<p><b>Result metrics:</b> {escape(str(metrics))}</p>"
        ), output))
        self.completed_cards = (*self.completed_cards, card)
        self.active_card = None
        self.status = "stage_complete"
        self._line(f"Completed {result.stage}: {receipt.approved_tool_call}; {receipt.scientific_label}: {receipt.scientific_invocation}; metrics={metrics}")
        self._set_body()

    def _request_synthesis(self) -> None:
        self.status = "synthesizing"
        try:
            result = self.controller.request_synthesis()
            output = widgets.Output()
            with output:
                demo_agent._display_conclusion(result)
            self.active_card = widgets.VBox((widgets.HTML("<h3>Final synthesis</h3>"), output))
            self.status = "complete"
            self._line("Final synthesis complete")
            self._set_body()
        except Exception as error:
            self._error_card("Synthesis failed", error, "Retry Synthesis", self._request_synthesis)


def launch_interactive_workflow(user_goal: str, api_key: str, skill: str | None = None,
                                client: Any = None, executors: dict[str, Any] | None = None) -> InteractiveWorkflow:
    controller = demo_agent.BoundedWorkflowController.create(
        user_goal, api_key, skill=skill, client=client, executors=executors
    )
    workflow = InteractiveWorkflow(controller)
    workflow.display()
    return workflow
