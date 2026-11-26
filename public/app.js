"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  mode: "recorded", role: "customer", session: false, conversation: null,
  busy: false, stopped: false, recording: null, step: 0, playing: false,
  timer: null, offerRecordings: false, refunds: new Map(), seenEvents: new Set(), cost: 0n,
};
const statuses = {
  SUCCEEDED: ["refunded", "success"],
  PENDING_APPROVAL: ["awaiting human review", "review"],
  PENDING: ["processing", "processing"],
  FAILED: ["declined", "failed"],
  REJECTED: ["rejected", "failed"],
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function notice(text, target = "notice") {
  $(target).textContent = text;
  $(target).hidden = !text;
}

function costText(nano) {
  const micros = (BigInt(nano || 0) + 500n) / 1000n;
  return `$${micros / 1000000n}.${String(micros % 1000000n).padStart(6, "0")}`;
}

function statusChip(status, prefix = "") {
  const [label, tone] = statuses[status] || ["status unavailable", "processing"];
  return element("span", prefix + label, `chip ${tone}`);
}

function syncControls() {
  const live = state.mode === "live";
  $("mode").textContent = live ? "Live" : "Recorded run";
  $("chat-context").textContent = live ? $("customer").selectedOptions[0].textContent : (state.recording?.customer || "Recorded run");
  $("recordings").hidden = live && !state.offerRecordings;
  $("customer-panel").hidden = state.role !== "customer";
  $("ops-panel").hidden = state.role !== "ops";
  $("role").value = state.role;
  for (const role of ["customer", "ops"]) {
    const tab = $(`${role}-tab`);
    tab.setAttribute("aria-selected", String(state.role === role));
    tab.tabIndex = state.role === role ? 0 : -1;
  }
  // Freeze navigation while a request is outstanding so results cannot cross views.
  for (const button of document.querySelectorAll("button")) button.disabled = state.busy;
  $("customer").disabled = state.busy;
  $("role").disabled = state.busy;
  $("message").disabled = !live || !state.conversation || state.stopped || state.busy;
  $("send").disabled = $("message").disabled || !$("message").value.trim();
  const finished = !state.recording || state.step >= recordingSteps().length;
  $("next").disabled = state.busy || finished || state.playing;
  $("play").disabled = state.busy || finished;
  $("play").textContent = state.playing ? "Pause" : "Play";
}

function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  state.timer = null;
}

function clearConversation() {
  pause();
  state.refunds.clear();
  state.seenEvents.clear();
  state.cost = 0n;
  state.step = 0;
  $("messages").replaceChildren(element("p", "Pick a recorded run or start a live demo to follow a refund.", "empty"));
  $("events").replaceChildren();
  $("events-empty").hidden = false;
  $("total-cost").textContent = costText(0n);
  $("refund-statuses").replaceChildren();
  $("refund-panel").hidden = true;
  $("decision-results").replaceChildren();
  $("message").value = "";
  updateCounter();
  notice("", "turn-notice");
}

function resetSession() {
  state.session = false;
  state.conversation = null;
  state.mode = "recorded";
  state.role = "customer";
  state.recording = null;
  state.stopped = false;
  state.offerRecordings = false;
  clearConversation();
  $("ops-queue").replaceChildren();
  $("escalations").replaceChildren();
  $("progress").textContent = "Choose a recorded run";
  for (const button of $("recording-list").querySelectorAll("button")) button.setAttribute("aria-pressed", "false");
}

class ApiError extends Error {
  constructor(status, data) {
    super("Demo request failed");
    this.status = status;
    this.data = data;
  }
}

async function api(path, method = "GET", body) {
  const options = { method, credentials: "same-origin", headers: { Accept: "application/json" } };
  if (method === "POST") {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body || {});
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new ApiError(response.status, data);
  return data;
}

function showError(error) {
  if (error.status === 401) {
    resetSession();
    notice("Your demo session expired. Start again.");
  } else if (error.status === 403) {
    notice("Switch role to use this tab.");
  } else if (error.status === 409) {
    notice("This request is already being processed or reviewed. Refresh and try again.");
  } else {
    notice("The demo is temporarily unavailable. Please try again.");
  }
}

async function action(button, loading, callback) {
  if (state.busy) return;
  pause();
  state.busy = true;
  const original = button ? button.textContent : "";
  syncControls();
  if (button) {
    button.textContent = loading;
    button.setAttribute("aria-busy", "true");
  }
  notice("");
  try {
    await callback();
  } catch (error) {
    showError(error);
  } finally {
    state.busy = false;
    if (button) {
      button.textContent = original;
      button.removeAttribute("aria-busy");
    }
    syncControls();
  }
}

function updateCounter() {
  $("counter").textContent = `${$("message").value.length.toLocaleString("en-US")} / 1,000`;
}

function appendMessage(role, text, refunds = []) {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  const wrapper = element("article", undefined, `message ${role}`);
  wrapper.append(element("span", role === "customer" ? "Customer" : "Agent · model-written", "caption"));
  const lines = String(text || "").split(/\r?\n/);
  const headers = ["Refund system status:", "حالة الاسترداد من النظام:"];
  const split = role === "agent" && refunds.length ? lines.findLastIndex((line) => headers.includes(line.trim())) : -1;
  const bubbleText = split < 0 ? text : lines.slice(0, split).join("\n").trimEnd();
  if (bubbleText) {
    const bubble = element("p", bubbleText, "bubble");
    bubble.dir = "auto";
    wrapper.append(bubble);
  }
  $("messages").append(wrapper);
  if (split >= 0) {
    const system = element("section", undefined, "system-output");
    system.append(element("span", "Refund system output at this step", "caption"));
    const output = element("p", lines.slice(split).join("\n"));
    output.dir = "auto";
    system.append(output);
    $("messages").append(system);
  }
  if (refunds.length) {
    const chips = element("div", undefined, "chip-list");
    for (const refund of refunds) {
      const chip = statusChip(refund.status, `Refund #${refund.refund_id} · `);
      chip.dataset.refundId = String(refund.refund_id);
      chips.append(chip);
    }
    $("messages").append(chips);
  }
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrapper;
}

function mergeRefunds(refunds, replace = false) {
  if (replace) state.refunds.clear();
  for (const refund of refunds || []) {
    const key = String(refund.refund_id);
    state.refunds.set(key, { ...state.refunds.get(key), ...refund });
    for (const chip of $("messages").querySelectorAll("[data-refund-id]")) {
      if (chip.dataset.refundId === key) {
        const fresh = statusChip(refund.status, `Refund #${key} · `);
        chip.textContent = fresh.textContent;
        chip.className = fresh.className;
      }
    }
  }
  $("refund-statuses").replaceChildren();
  for (const refund of state.refunds.values()) {
    const prefix = [
      `Refund #${refund.refund_id}`,
      refund.order_id !== undefined ? `Order #${refund.order_id}` : "",
      refund.amount !== undefined ? `$${refund.amount}` : "",
    ].filter(Boolean).join(" · ") + " · ";
    $("refund-statuses").append(statusChip(refund.status, prefix));
  }
  $("refund-panel").hidden = state.refunds.size === 0;
}

function eventSummary(event) {
  const data = event.data || {};
  switch (event.type) {
    case "customer_message": return "Customer message received";
    case "model_call": {
      const usage = data.usage || {};
      const input = (usage.input || 0) + (usage.cache_read || 0) + (usage.cache_write || 0);
      return `Model call · ${input.toLocaleString("en-US")} in / ${(usage.output || 0).toLocaleString("en-US")} out · ${costText(data.cost_nano)}`;
    }
    case "tool_call": return `Tool → ${data.name}\n${JSON.stringify(data.input || {})}`;
    case "policy_decision": return `Policy → ${data.outcome} (${(data.reasons || []).join(", ")})\nFacts: ${JSON.stringify(data.facts || {})}`;
    case "dops_call": return `Refund system · ${data.operation} · ${data.status || "requested"}`;
    case "refund_result": return `Refund #${data.refund_id} · ${data.status}${data.replayed ? " · same intent replayed" : ""}`;
    case "approval": return `Human approved refund #${data.refund_id} · ${data.status}`;
    case "rejection": return `Human rejected refund #${data.refund_id} · ${data.status}`;
    case "approval_requested": return `Human requested approval · Refund #${data.refund_id}`;
    case "rejection_requested": return `Human requested rejection · Refund #${data.refund_id}`;
    case "escalation": return "Escalation opened for human review";
    case "incident": return `System incident · ${data.kind || "temporarily unavailable"}`;
    case "budget_refused": return "Demo budget reached · model call refused";
    default: return String(event.type).replaceAll("_", " ");
  }
}

function appendEvents(events) {
  for (const event of events || []) {
    const key = JSON.stringify(event);
    if (state.seenEvents.has(key)) continue;
    state.seenEvents.add(key);
    const row = element("li", undefined, "event");
    const meta = element("div", undefined, "event-meta");
    const date = new Date(event.ts);
    const time = element("time", Number.isNaN(date.getTime()) ? "Time unavailable" : date.toLocaleTimeString("en-GB", { hour12: false }));
    if (!Number.isNaN(date.getTime())) time.dateTime = event.ts;
    meta.append(time, element("span", event.actor, "actor"));
    if (event.trace_id) meta.append(element("span", event.trace_id));
    row.append(meta, element("p", eventSummary(event), "event-summary"));
    $("events").append(row);
    if (event.type === "model_call") state.cost += BigInt(event.data.cost_nano || 0);
  }
  $("events-empty").hidden = state.seenEvents.size > 0;
  $("total-cost").textContent = costText(state.cost);
}

function recordingSteps() {
  if (!state.recording) return [];
  const steps = [...state.recording.steps];
  if (state.recording.ops?.length) steps.push({ role: "ops", actions: state.recording.ops });
  return steps;
}

function nextStep() {
  const steps = recordingSteps();
  if (state.step >= steps.length) return;
  const step = steps[state.step++];
  if (step.role === "ops") {
    for (const decision of step.actions) {
      const box = element("section", undefined, "system-output");
      const verb = decision.action === "approve" ? "approved" : "rejected";
      const label = statuses[decision.status_after]?.[0] || "status unavailable";
      box.append(element("p", `Ops ${verb} refund → ${label}`), statusChip(decision.status_after, `Refund #${decision.refund_id} · `));
      $("messages").append(box);
      mergeRefunds([{ refund_id: decision.refund_id, status: decision.status_after }]);
    }
  } else {
    appendMessage(step.role, step.text, step.refunds || []);
    mergeRefunds(step.refunds);
    appendEvents(step.events);
  }
  $("progress").textContent = `Step ${state.step} of ${steps.length}${state.step === steps.length ? " · Complete" : ""}`;
  $("messages").scrollTop = $("messages").scrollHeight;
  if (state.step === steps.length) pause();
  syncControls();
}

function play() {
  if (state.playing) {
    pause();
    syncControls();
    return;
  }
  state.playing = true;
  const tick = () => {
    if (!state.playing) return;
    nextStep();
    if (state.playing) state.timer = setTimeout(tick, 1200);
  };
  tick();
}

async function loadRecordings() {
  let recordings;
  try {
    recordings = await api("/api/recordings");
  } catch (error) {
    const retry = element("button", "Retry recorded runs", "secondary");
    retry.type = "button";
    retry.addEventListener("click", () => action(retry, "Loading\u2026", loadRecordings));
    $("recording-list").replaceChildren(retry);
    throw error;
  }
  $("recording-list").replaceChildren();
  $("recording-list").setAttribute("aria-busy", "false");
  if (!recordings.length) {
    $("recording-list").append(element("p", "No recorded runs yet. Start a live demo to try it.", "empty"));
    return;
  }
  for (const recording of recordings) {
    const button = element("button", recording.title, "secondary");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => action(button, "Loading…", async () => {
      const run = await api(`/api/recordings/${encodeURIComponent(recording.name)}`);
      clearConversation();
      state.mode = "recorded";
      state.conversation = null;
      state.recording = run;
      for (const item of $("recording-list").querySelectorAll("button")) item.setAttribute("aria-pressed", String(item === button));
      $("progress").textContent = `Step 0 of ${recordingSteps().length}`;
      $("chat-context").textContent = run.customer;
    }));
    $("recording-list").append(button);
  }
}

async function createSession() {
  const result = await api("/api/session", "POST", { customer_id: Number($("customer").value) });
  state.session = true;
  state.role = "customer";
  return result;
}

async function startLive() {
  await action($("start-live"), "Starting…", async () => {
    resetSession();
    try {
      const session = await createSession();
      const result = await api("/api/live/start", "POST", {});
      clearConversation();
      state.mode = "live";
      state.conversation = result.conversation_id;
      state.recording = null;
      state.stopped = false;
      $("messages").replaceChildren(element("p", `${session.customer.name}: ask about a synthetic order or request a refund.`, "empty"));
      const orders = element("section", undefined, "system-output");
      orders.append(element("h4", "Your synthetic orders · refund system facts"));
      for (const order of session.orders) {
        orders.append(element("p", `Order #${order.id} · ${order.status} · $${order.total_amount} · reserved $${order.reserved_amount} · refunded $${order.refunded_amount}`));
      }
      $("messages").append(orders);
      notice("Live demo started. Messages may call the model within the demo budget.");
    } catch (error) {
      if (error.status !== 429) throw error;
      resetSession();
      notice(error.data.reason === "daily_budget"
        ? "Today's demo budget is used up. Showing recorded runs instead."
        : "You've reached today's live demo limit. Showing recorded runs instead.");
    }
  });
  if (state.mode === "live") $("message").focus();
}

async function refreshRefunds() {
  if (!state.conversation) return;
  const refunds = await api(`/api/conversation/${encodeURIComponent(state.conversation)}/refunds`);
  mergeRefunds(refunds, true);
}

async function refreshAudit() {
  if (!state.conversation) return;
  appendEvents(await api(`/api/audit?conversation_id=${encodeURIComponent(state.conversation)}`));
}

const stopNotices = {
  budget: "This conversation reached the demo budget. Recorded runs are available below.",
  turn_limit: "This conversation reached its turn limit. Start a new live demo to continue.",
  busy: "A reply is already in progress. Please try again shortly.",
  model_error: "The model could not finish this reply. Please try again.",
  step_limit: "This request reached its step limit. Please try a shorter request.",
  refusal: "The model could not answer this request. You can try another message.",
};

async function sendMessage(event) {
  event.preventDefault();
  const text = $("message").value.trim();
  if (state.busy || !state.conversation || state.stopped || !text || text.length > 1000) return;
  await action($("send"), "Sending…", async () => {
    const customer = appendMessage("customer", text);
    $("message").value = "";
    updateCounter();
    const typing = element("p", "Agent is typing…", "caption typing");
    typing.setAttribute("role", "status");
    $("messages").append(typing);
    notice("", "turn-notice");
    let result;
    try {
      result = await api("/api/chat", "POST", { conversation_id: state.conversation, message: text });
    } catch (error) {
      if (error.status === 409) {
        customer.append(element("span", "Not sent · conversation busy", "caption"));
        $("message").value = text;
        updateCounter();
        notice(stopNotices.busy, "turn-notice");
        await refreshRefunds();
        await refreshAudit();
        return;
      }
      customer.append(element("span", "Delivery not confirmed", "caption"));
      $("message").value = text;
      updateCounter();
      throw error;
    } finally {
      typing.remove();
    }
    appendMessage("agent", result.reply, result.refunds || []);
    mergeRefunds(result.refunds);
    appendEvents(result.events);
    state.stopped = ["budget", "turn_limit"].includes(result.stopped);
    notice(stopNotices[result.stopped] || "", "turn-notice");
    state.offerRecordings = result.stopped === "budget";
    await refreshRefunds();
  });
  if (!$("message").disabled) $("message").focus();
}

function modelReason(text, title = "Model-written reason") {
  const box = element("section", undefined, "model-reason");
  box.append(element("h4", title), element("span", "model-written · untrusted context", "caption"));
  const content = element("p", text);
  content.dir = "auto";
  box.append(content);
  return box;
}

function renderQueue(queue) {
  $("ops-queue").replaceChildren();
  if (!queue.refunds.length) $("ops-queue").append(element("p", "No refunds are waiting for review.", "empty"));
  for (const refund of queue.refunds) {
    const card = element("article", undefined, "ops-card");
    card.append(element("h3", `Refund #${refund.refund_id}`), statusChip("PENDING_APPROVAL"));
    const facts = element("section", undefined, "facts");
    facts.append(element("h4", "Facts (from the refund system)"));
    const list = element("dl");
    const data = refund.facts;
    for (const [label, value] of [
      ["Amount", `$${data.amount}`], ["Order total", `$${data.order_total}`],
      ["Reserved", `$${data.reserved}`], ["Already refunded", `$${data.refunded}`],
      ["Customer", `#${data.customer_id}`], ["Order", `#${data.order_id}`],
    ]) list.append(element("dt", label), element("dd", value));
    facts.append(list, element("h4", "Policy reasons"));
    const reasons = element("div", undefined, "chip-list");
    for (const reason of data.policy_reasons) reasons.append(element("span", reason, "chip review reason-chip"));
    if (!data.policy_reasons.length) reasons.append(element("span", "Held by the refund system", "chip review"));
    facts.append(reasons);
    card.append(facts, modelReason(refund.model_text.text));
    const form = element("form", undefined, "decision-form");
    const reasonId = `reason-${refund.refund_id}`;
    const label = element("label", "Rejection reason (required to reject)");
    label.htmlFor = reasonId;
    const reason = element("textarea");
    reason.id = reasonId;
    reason.dir = "auto";
    reason.maxLength = 500;
    reason.rows = 2;
    reason.required = true;
    const actions = element("div", undefined, "actions");
    const approve = element("button", "Approve");
    approve.type = "button";
    approve.addEventListener("click", () => decide(refund.refund_id, "approve", "", approve));
    const reject = element("button", "Reject", "reject");
    reject.type = "submit";
    reason.addEventListener("input", () => reason.setCustomValidity(""));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!reason.value.trim()) {
        reason.setCustomValidity("Enter a rejection reason.");
        reason.reportValidity();
        return;
      }
      decide(refund.refund_id, "reject", reason.value.trim(), reject);
    });
    actions.append(approve, reject);
    form.append(label, reason, actions);
    card.append(form);
    $("ops-queue").append(card);
  }
  if (queue.unconfirmed?.length || queue.unconfirmed_remaining) {
    const section = element("section", undefined, "ops-card");
    section.append(element("h3", "Unconfirmed refund requests"));
    const list = element("ul");
    for (const row of queue.unconfirmed) {
      list.append(element("li", `Order #${row.order_id} · $${row.amount}`));
    }
    section.append(list);
    if (queue.unconfirmed_remaining) {
      section.append(element("p", `${queue.unconfirmed_remaining} more unconfirmed requests`));
    }
    $("ops-queue").append(section);
  }
  $("escalations").replaceChildren();
  if (!queue.escalations.length) $("escalations").append(element("p", "No escalations need attention.", "empty"));
  for (const escalation of queue.escalations) {
    const card = element("article", undefined, "escalation-card");
    card.append(element("h4", `Customer #${escalation.customer_id} · ${escalation.status}`));
    card.append(modelReason(escalation.model_text.text, "Model-written summary"));
    $("escalations").append(card);
  }
}

async function loadQueue() {
  renderQueue(await api("/api/ops/queue"));
}

async function decide(id, decision, reason, button) {
  await action(button, decision === "approve" ? "Approving…" : "Rejecting…", async () => {
    let result;
    try {
      result = decision === "approve"
        ? await api(`/api/ops/refunds/${id}/approve`, "POST", {})
        : await api(`/api/ops/refunds/${id}/reject`, "POST", { reason });
    } catch (error) {
      if (error.status === 409) await loadQueue();
      throw error;
    }
    const output = element("p", `Refund #${id} · Human decision recorded: `, "decision-result");
    output.append(statusChip(result.status));
    $("decision-results").append(output);
    await loadQueue();
    await refreshAudit();
  });
}

async function changeRole(role, button) {
  if (state.busy || role === state.role) return;
  await action(button, "Switching…", async () => {
    if (!state.session) await createSession();
    await api("/api/session/role", "POST", { role });
    state.role = role;
    if (role === "ops") await loadQueue();
    else {
      await refreshRefunds();
      await refreshAudit();
    }
  });
}

$("start-live").addEventListener("click", startLive);
$("next").addEventListener("click", nextStep);
$("play").addEventListener("click", play);
$("customer").addEventListener("change", () => {
  resetSession();
  syncControls();
});
$("role").addEventListener("change", () => changeRole($("role").value, $(`${$("role").value}-tab`)));
for (const role of ["customer", "ops"]) {
  $(`${role}-tab`).addEventListener("click", () => changeRole(role, $(`${role}-tab`)));
  $(`${role}-tab`).addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? "customer" : event.key === "End" ? "ops" : role === "customer" ? "ops" : "customer";
    $(`${next}-tab`).focus();
    changeRole(next, $(`${next}-tab`));
  });
}
$("refresh-ops").addEventListener("click", () => action($("refresh-ops"), "Refreshing…", loadQueue));
$("chat-form").addEventListener("submit", sendMessage);
$("message").addEventListener("input", () => {
  updateCounter();
  syncControls();
});
$("message").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!$("send").disabled) $("chat-form").requestSubmit();
  }
});
if (window.matchMedia("(max-width: 1023px)").matches) $("behind").open = false;
window.addEventListener("pagehide", pause);
action(null, "", loadRecordings).then(() => {
  $("recording-list").setAttribute("aria-busy", "false");
});
