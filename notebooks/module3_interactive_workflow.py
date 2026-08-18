"""Interactive notebook presentation for the bounded Module 3 panel-design agent."""

from __future__ import annotations

from html import escape
import re
from typing import Any, Callable

import ipywidgets as widgets
from IPython.display import display as ipython_display

from workshop_llm_agent import PanelAgentRun, PanelDesignAgent, PanelPlan


MODULE3_WORKFLOW_VERSION = "2026.08.18.2"


def _safe_text(value: Any, limit: int = 6000) -> str:
    """Keep notebook messages readable and avoid displaying a hosted API key."""
    text = re.sub(r"nvapi-[A-Za-z0-9_-]+", "nvapi-[hidden]", str(value))
    return text[-limit:]


class InteractivePanelDesignWorkflow:
    """Plan, approve, execute, and review one panel-design run in stage cards."""

    def __init__(
        self,
        agent: PanelDesignAgent,
        *,
        expected_panel_size: int,
        max_revisions: int = 0,
        timeout_seconds: int = 480,
        on_complete: Callable[[PanelAgentRun], None] | None = None,
    ) -> None:
        self.agent = agent
        self.expected_panel_size = expected_panel_size
        self.status = "idle"
        self.plan: PanelPlan | None = None
        self.agent_run: PanelAgentRun | None = None
        self.transcript_text = ""
        self._busy = False
        self._on_complete = on_complete
        self._completion_callback_called = False

        self.start_button = widgets.Button(
            description="Start Agent", button_style="primary", icon="play"
        )
        self.start_button.on_click(self._start)
        # The full titles already appear in strategy cards. Short radio labels avoid
        # frontend-dependent wrapping that can make adjacent choices overlap.
        self.strategy_control = widgets.RadioButtons(
            description="Choose",
            layout=widgets.Layout(width="280px"),
            style={"description_width": "70px"},
        )
        if type(max_revisions) is not int or max_revisions != 0:
            raise ValueError("max_revisions must be 0 for controller-owned source.")
        self.timeout_control = widgets.Dropdown(
            options=(("5 minutes", 300), ("8 minutes", 480), ("10 minutes", 600)),
            value=timeout_seconds,
            description="Limit",
        )
        self.approve_button = widgets.Button(
            description="Approve Plan & Run Agent",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="280px"),
        )
        self.approve_button.on_click(self._approve_and_run)
        self.approve_button.disabled = True

        self._cards = widgets.VBox()
        self.result_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #b8b8b8", padding="8px")
        )
        self.root = widgets.VBox((self.start_button, self._cards, self.result_output))

    def display(self) -> widgets.VBox:
        ipython_display(self.root)
        return self.root

    def _line(self, message: str) -> None:
        self.transcript_text += message + "\n"

    def _append(self, widget: widgets.Widget) -> None:
        self._cards.children = (*self._cards.children, widget)

    @staticmethod
    def _html_card(title: str, body: str) -> widgets.VBox:
        return widgets.VBox(
            (
                widgets.HTML(
                    "<div style='border-left:5px solid #76b900; padding:8px 14px; "
                    "margin:6px 0; background:#f7f7f7'>"
                    f"<h3>{escape(title)}</h3>{body}</div>"
                ),
            )
        )

    def _error_card(self, title: str, error: Exception) -> None:
        message = _safe_text(f"{type(error).__name__}: {error}")
        self._line(f"{title}: {message}")
        self._append(self._html_card(title, f"<pre>{escape(message)}</pre>"))

    def _start(self, button: widgets.Button) -> None:
        if self._busy or self.status != "idle" or button is not self.start_button:
            return
        self._busy = True
        self.status = "planning"
        self.start_button.disabled = True
        self._append(
            self._html_card(
                "Planning",
                "<p>Nemotron is inspecting the bounded data profile and proposing two "
                "panel-design strategies. No analysis code is generated yet.</p>",
            )
        )
        try:
            self.plan = self.agent.request_plan()
            self._show_plan(self.plan)
            self.status = "awaiting_approval"
        except Exception as error:
            self.status = "plan_failed"
            self._error_card("Plan request failed", error)
            retry = widgets.Button(description="Retry Plan", icon="refresh")
            retry.on_click(self._retry_plan)
            self._append(retry)
        finally:
            self._busy = False

    def _retry_plan(self, button: widgets.Button) -> None:
        if self._busy or self.status != "plan_failed":
            return
        button.disabled = True
        self.status = "idle"
        self.start_button.disabled = False
        self._start(self.start_button)

    def _show_plan(self, plan: PanelPlan) -> None:
        observations = "".join(
            f"<li>{escape(item)}</li>" for item in plan.data_observations
        )
        self._append(self._html_card("Agent observations", f"<ul>{observations}</ul>"))

        options = []
        for index, strategy in enumerate(plan.strategies, start=1):
            options.append((f"Strategy {index}", index))
            body = (
                f"<p><b>Approach:</b> {escape(strategy.approach)}</p>"
                f"<p><b>Property coverage:</b> {escape(strategy.property_coverage_measure)}</p>"
                f"<p><b>Cluster balance:</b> {escape(strategy.cluster_balance)}</p>"
                f"<p><b>Tradeoff:</b> {escape(strategy.tradeoff)}</p>"
            )
            self._append(self._html_card(f"Strategy {index}: {strategy.title}", body))

        self.strategy_control.options = tuple(options)
        self.strategy_control.value = plan.recommended_strategy
        recommendation = widgets.HTML(
            f"<p><b>Agent recommendation:</b> strategy {plan.recommended_strategy}. "
            f"{escape(plan.recommendation_reason)}</p>"
        )
        approval = widgets.VBox(
            (
                recommendation,
                self.strategy_control,
                self.timeout_control,
                self.approve_button,
            )
        )
        self.approve_button.disabled = False
        self._append(approval)
        self._line(
            f"Plan ready; agent recommended strategy {plan.recommended_strategy}."
        )

    def _approve_and_run(self, button: widgets.Button) -> None:
        if (
            self._busy
            or self.status != "awaiting_approval"
            or button is not self.approve_button
            or button.disabled
            or self.plan is None
        ):
            return
        self._busy = True
        self.status = "executing"
        self.approve_button.disabled = True
        self.strategy_control.disabled = True
        self.timeout_control.disabled = True
        approved_strategy = int(self.strategy_control.value)
        self._line(f"Sponsor approved strategy {approved_strategy}.")
        try:
            self.agent_run = self.agent.run(
                approved_strategy=approved_strategy,
                expected_panel_size=self.expected_panel_size,
                max_revisions=0,
                timeout_seconds=int(self.timeout_control.value),
                progress_callback=self._progress,
            )
            self.status = "completed" if self.agent_run.success else "failed"
            self._show_final_result(self.agent_run)
            if self.agent_run.success and not self._completion_callback_called:
                self._completion_callback_called = True
                if self._on_complete is not None:
                    with self.result_output:
                        try:
                            self._on_complete(self.agent_run)
                        except Exception as error:
                            message = _safe_text(f"{type(error).__name__}: {error}")
                            self._line(f"Completion display failed: {message}")
                            self._append(
                                self._html_card(
                                    "Completion display failed",
                                    "<p>The validated scientific result is unchanged.</p>"
                                    f"<pre>{escape(message)}</pre>",
                                )
                            )
        except Exception as error:
            self.status = "failed"
            self._error_card("Agent run stopped safely", error)
        finally:
            self._busy = False

    def _progress(self, event: str, payload: dict[str, Any]) -> None:
        attempt = payload.get("attempt")
        if event == "run_started":
            self._append(
                self._html_card(
                    "Approved run",
                    f"<p>Strategy {payload['approved_strategy']} is approved. The local "
                    "controller will render its tested nvMolKit implementation. "
                    f"The controller expects exactly {payload['expected_panel_size']} compounds "
                    "and performs one bounded execution.</p>",
                )
            )
        elif event == "source_rendered":
            tradeoffs = "".join(
                f"<li>{escape(item)}</li>"
                for item in payload.get("expected_tradeoffs", ())
            )
            code = widgets.HTML(
                "<pre style='max-height:420px; overflow:auto; white-space:pre; "
                "border:1px solid #bbb; padding:10px'>"
                f"{escape(payload['source'])}</pre>"
            )
            accordion = widgets.Accordion(children=(code,))
            accordion.set_title(0, f"View rendered {payload['source_file']}")
            summary = self._html_card(
                f"Attempt {attempt}: source rendered",
                f"<p>{escape(payload['implementation_summary'])}</p>"
                f"<p><b>Expected tradeoffs:</b></p><ul>{tradeoffs}</ul>"
                "<p>The executable source is preserved before validation and execution.</p>",
            )
            self._append(widgets.VBox((summary, accordion)))
            self._line(f"Rendered and preserved {payload['source_file']}.")
        elif event == "source_validated":
            self._append(
                self._html_card(
                    f"Attempt {attempt}: source passed static checks",
                    "<p>The exact controller script satisfies the bounded source contract. "
                    "The controller will run it without the hosted API key and validate "
                    "its artifacts independently.</p>",
                )
            )
            self._line(f"Statically validated {payload['source_file']}.")
        elif event == "execution_started":
            self._append(
                self._html_card(
                    f"Attempt {attempt}: executing",
                    f"<p>The bounded analysis is running with a {payload['timeout_seconds']}-second limit.</p>",
                )
            )
        elif event == "attempt_passed":
            receipt = escape(str(payload["receipt"]))
            self._append(
                self._html_card(
                    f"Attempt {attempt}: validated",
                    f"<p>Completed in {payload['elapsed_seconds']:.2f} seconds.</p>"
                    f"<pre>{receipt}</pre>",
                )
            )
            self._line(f"Attempt {attempt} passed artifact validation.")
        elif event == "attempt_failed":
            action = (
                "The source is controller-owned, so this receipt indicates an environment, "
                "runtime, or controller defect rather than another prompt-repair task."
            )
            self._append(
                self._html_card(
                    f"Attempt {attempt}: validation failed",
                    f"<pre>{escape(_safe_text(payload['message']))}</pre><p>{escape(action)}</p>",
                )
            )
            self._line(
                f"Attempt {attempt} failed; revise={payload.get('will_revise')}."
            )
        elif event == "audit_started":
            self._append(
                self._html_card(
                    "Scientific audit",
                    "<p>Validated artifacts are being reviewed for tradeoffs, boundaries, "
                    "and a defensible next iteration.</p>",
                )
            )
        elif event == "audit_completed":
            audit = payload["audit"]
            self._line("Analysis validated; audit complete")
            body = (
                f"<p><b>Assessment:</b> {escape(audit['result_assessment'])}</p>"
                f"<p><b>Important observation:</b> {escape(audit['surprising_result'])}</p>"
                f"<p><b>Scientific boundaries:</b> {escape(audit['scientific_boundaries'])}</p>"
                f"<p><b>Next iteration:</b> {escape(audit['next_iteration'])}</p>"
            )
            self._append(self._html_card("Audit complete", body))
        elif event == "audit_failed":
            self._line("Analysis validated; audit unavailable")
            self._append(
                self._html_card(
                    "Optional audit unavailable",
                    f"<pre>{escape(_safe_text(payload['message']))}</pre>",
                )
            )

    def _show_final_result(self, run: PanelAgentRun) -> None:
        if run.success:
            status = (
                "Analysis validated; audit complete"
                if run.audit is not None
                else "Analysis validated; audit unavailable"
            )
            body = (
                "<p><b>Success:</b> the rendered analysis produced artifacts that passed "
                "independent validation.</p>"
                f"<ul><li>Analysis: <code>{escape(str(run.analysis_path))}</code></li>"
                f"<li>Panel: <code>{escape(str(run.panel_path))}</code></li>"
                f"<li>Report: <code>{escape(str(run.report_path))}</code></li>"
                f"<li>Trace: <code>{escape(str(run.trace_path))}</code></li></ul>"
                "<p>The shared completion renderer can now display these validated artifacts.</p>"
            )
            self._line(status)
            self._append(self._html_card(status, body))
        else:
            self._append(
                self._html_card(
                    "Agent workflow did not pass",
                    f"<p>All bounded attempts were used. Inspect <code>{escape(str(run.trace_path))}</code>. "
                    "No panel is available until a validated run succeeds.</p>",
                )
            )


def launch_interactive_panel_design(
    agent: PanelDesignAgent,
    *,
    expected_panel_size: int,
    max_revisions: int = 0,
    timeout_seconds: int = 480,
    on_complete: Callable[[PanelAgentRun], None] | None = None,
) -> InteractivePanelDesignWorkflow:
    """Display and return the Module 3 interactive workflow."""
    workflow = InteractivePanelDesignWorkflow(
        agent,
        expected_panel_size=expected_panel_size,
        max_revisions=max_revisions,
        timeout_seconds=timeout_seconds,
        on_complete=on_complete,
    )
    workflow.display()
    workflow._start(workflow.start_button)
    return workflow
