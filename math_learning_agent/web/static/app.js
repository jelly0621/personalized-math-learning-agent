const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value = "") => String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const tags = (items = []) => `<span class="tags">${items.map(x => `<span class="tag">${escapeHTML(x)}</span>`).join("")}</span>`;
const dateText = value => value ? new Date(value.replace(" ", "T") + (value.includes("Z") ? "" : "Z")).toLocaleString("zh-CN") : "--";
let currentQuestionId = null;

async function api(path, options = {}) {
  const defaultHeaders = options.body instanceof FormData ? {} : {"Content-Type": "application/json"};
  const response = await fetch(path, {headers: defaultHeaders, ...options});
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}
function toast(message, error = false) {
  const el = $("#toast"); el.textContent = message; el.className = `show${error ? " error-toast" : ""}`;
  clearTimeout(window.toastTimer); window.toastTimer = setTimeout(() => el.className = "", 4200);
}
function setBusy(button, busy, text = "模型处理中，请稍候…") {
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.idle;
}
function optional(value) { const text = String(value || "").trim(); return text || null; }

async function loadHealth() {
  try { const data = await api("/api/health"); $("#health-badge").textContent = data.status === "ok" ? "服务正常" : "服务异常"; $("#health-badge").className = "badge success"; }
  catch { $("#health-badge").textContent = "服务不可用"; $("#health-badge").className = "badge danger"; }
}
function profileTable(rows, compact = false) {
  if (!rows.length) return `<div class="empty">暂无学生画像数据</div>`;
  const shown = compact ? rows.slice(0, 5) : rows;
  return `<table><thead><tr><th>知识点</th><th>错题</th><th>练习</th><th>正确</th><th>训练正确率</th>${compact ? "" : "<th>最近错题</th><th>最近练习</th>"}</tr></thead><tbody>${shown.map(row => {
    const accuracy = row.practice_count ? `${Math.round(row.correct_count / row.practice_count * 100)}%` : "--";
    return `<tr><td>${escapeHTML(row.knowledge_point)}</td><td>${row.wrong_count}</td><td>${row.practice_count}</td><td>${row.correct_count}</td><td>${accuracy}</td>${compact ? "" : `<td>${dateText(row.last_wrong_at)}</td><td>${dateText(row.last_practiced_at)}</td>`}</tr>`;
  }).join("")}</tbody></table>`;
}
async function loadProfile() {
  try { const rows = await api("/api/profile"); $("#profile-table").innerHTML = profileTable(rows); $("#overview-profile").innerHTML = profileTable(rows, true); }
  catch (error) { $("#profile-table").innerHTML = `<p class="error">请求失败：${escapeHTML(error.message)}</p>`; }
}
function attemptItems(rows) {
  if (!rows.length) return `<div class="empty">暂无训练记录</div>`;
  return rows.map(row => `<div class="history-item"><span class="badge ${row.is_correct ? "success" : "warning"}">${row.is_correct ? "正确" : "需要加强"}</span><p>${escapeHTML(row.feedback)}</p><small>题目 #${row.question_id} · ${dateText(row.created_at)}</small></div>`).join("");
}
async function loadDashboard() {
  await loadProfile();
  try { const attempts = await api("/api/history/training-attempts?limit=5"); $("#overview-attempts").innerHTML = attemptItems(attempts); }
  catch (error) { $("#overview-attempts").innerHTML = `<p class="error">请求失败：${escapeHTML(error.message)}</p>`; }
}

let selectedImages = [];
let previewUrls = [];
let extractedProblems = [];

document.querySelectorAll(".intake-tab").forEach(tab => tab.addEventListener("click", () => {
  document.querySelectorAll(".intake-tab").forEach(item => item.classList.toggle("active", item === tab));
  $("#photo-intake").classList.toggle("hidden", tab.dataset.intake !== "photo");
  $("#manual-intake").classList.toggle("hidden", tab.dataset.intake !== "manual");
}));

function addImages(files) {
  const allowed = new Set(["image/jpeg", "image/png", "image/webp"]);
  for (const file of files) {
    if (selectedImages.length >= 5) { toast("一次最多选择 5 张图片", true); break; }
    if (!allowed.has(file.type)) { toast(`不支持的图片类型：${file.name}`, true); continue; }
    if (file.size > 10 * 1024 * 1024) { toast(`图片超过 10MB：${file.name}`, true); continue; }
    selectedImages.push(file);
  }
  renderImagePreviews();
}
function renderImagePreviews() {
  previewUrls.forEach(url => URL.revokeObjectURL(url)); previewUrls = [];
  if (!selectedImages.length) {
    $("#photo-previews").innerHTML = `<p class="empty">尚未选择图片</p>`;
  } else {
    $("#photo-previews").innerHTML = selectedImages.map((file, index) => {
      const url = URL.createObjectURL(file); previewUrls.push(url);
      return `<div class="preview-item"><img src="${url}" alt="图片 ${index + 1} 预览"><button type="button" class="remove-preview" data-index="${index}" aria-label="删除图片">×</button><small>${escapeHTML(file.name)}</small></div>`;
    }).join("");
    document.querySelectorAll(".remove-preview").forEach(button => button.addEventListener("click", () => {
      selectedImages.splice(Number(button.dataset.index), 1); renderImagePreviews();
    }));
  }
  $("#extract-photo-button").disabled = selectedImages.length === 0;
}
const photoInput = $("#photo-input");
photoInput.addEventListener("change", () => { addImages(photoInput.files); photoInput.value = ""; });
const dropzone = $("#photo-dropzone");
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", event => addImages(event.dataTransfer.files));

function updateImportButton() {
  const count = document.querySelectorAll(".problem-select:checked").length;
  const button = $("#import-photo-button");
  button.disabled = count === 0;
  button.dataset.idle = `分析并录入 ${count} 道错题`;
  if (!button.disabled) button.textContent = button.dataset.idle;
}
function renderExtractedProblems(warnings = []) {
  $("#extraction-confirmation").classList.toggle("hidden", extractedProblems.length === 0 && warnings.length === 0);
  $("#extraction-count").textContent = `识别到 ${extractedProblems.length} 道题`;
  $("#extracted-problems").innerHTML = extractedProblems.map((problem, index) => `<article class="card extracted-card" data-index="${index}"><button type="button" class="remove-extracted" data-index="${index}">删除</button><label class="problem-check"><input class="problem-select" type="checkbox" checked>作为不会做的错题录入 · 题目 ${problem.problem_index}</label><p class="muted">来源：图片 ${problem.source_image_index}</p><label>题目正文<textarea class="extracted-text" rows="5">${escapeHTML(problem.problem_text)}</textarea></label><label>教材答案<textarea class="extracted-answer" rows="2" placeholder="图片未明确展示时请留空">${escapeHTML(problem.printed_answer || "")}</textarea></label><details><summary>教材解析（可编辑）</summary><textarea class="extracted-solution" rows="5" placeholder="图片未明确展示时请留空">${escapeHTML(problem.printed_solution || "")}</textarea></details><div class="confidence-row"><span>识别置信度</span><strong>${Math.round(problem.confidence * 100)}%</strong></div></article>`).join("");
  $("#extraction-warnings").innerHTML = warnings.length ? `<div class="warning-list"><strong>识别提示</strong><ul>${warnings.map(warning => `<li>${escapeHTML(warning)}</li>`).join("")}</ul></div>` : "";
  document.querySelectorAll(".remove-extracted").forEach(button => button.addEventListener("click", () => { extractedProblems.splice(Number(button.dataset.index), 1); renderExtractedProblems(warnings); }));
  document.querySelectorAll(".problem-select").forEach(box => box.addEventListener("change", () => { box.closest(".extracted-card").classList.toggle("excluded", !box.checked); updateImportButton(); }));
  updateImportButton();
}

$("#extract-photo-button").addEventListener("click", async event => {
  const button = event.currentTarget; setBusy(button, true, "正在识别教材图片中的数学题，可能需要几十秒……");
  const form = new FormData(); selectedImages.forEach(image => form.append("images", image, image.name));
  try {
    const result = await api("/api/photo/extract", {method: "POST", body: form});
    extractedProblems = result.problems;
    renderExtractedProblems(result.warnings);
    toast(`识别到 ${result.problems.length} 道题，请确认后再录入`);
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); button.disabled = selectedImages.length === 0; }
});

$("#import-photo-button").addEventListener("click", async event => {
  const cards = [...document.querySelectorAll(".extracted-card")].filter(card => card.querySelector(".problem-select").checked);
  if (!cards.length) { toast("请至少选择一道题", true); return; }
  const problems = cards.map(card => ({
    problem_text: card.querySelector(".extracted-text").value.trim(),
    correct_answer: optional(card.querySelector(".extracted-answer").value),
    student_answer: null,
    student_solution: null,
  }));
  if (problems.some(problem => !problem.problem_text)) { toast("选中的题目正文不能为空", true); return; }
  const button = event.currentTarget; setBusy(button, true, "正在逐题进行题目理解与错因诊断，请稍候……");
  try {
    const result = await api("/api/photo/import", {method: "POST", body: JSON.stringify({problems})});
    $("#photo-batch-result").className = "card";
    $("#photo-batch-result").innerHTML = `<h3>录入完成</h3><div class="batch-summary"><span class="badge success">成功 ${result.succeeded}</span><span class="badge ${result.failed ? "danger" : "neutral"}">失败 ${result.failed}</span></div>` + result.results.map(item => `<details class="batch-item"><summary>题目 ${item.index} · ${item.status === "succeeded" ? "已保存" : "处理失败"}</summary>${item.status === "succeeded" ? `<p>${escapeHTML(item.problem_text)}</p><p>知识点：${tags(item.knowledge_points)}</p><p>题型：${escapeHTML(item.question_type)} · 难度：${item.difficulty}</p><p>摘要：${escapeHTML(item.problem_summary)}</p><p>错误类型：<span class="badge neutral">${escapeHTML(item.error_type)}</span></p><p>${escapeHTML(item.error_reason)}</p><p>相关知识点：${tags(item.related_knowledge_points)}</p><p>置信度：${Math.round(item.error_confidence * 100)}%</p>` : `<p class="error">${escapeHTML(item.error)}</p>`}</details>`).join("");
    toast(`批量录入完成：成功 ${result.succeeded}，失败 ${result.failed}`, result.failed > 0);
    await loadDashboard(); await loadHistory();
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); updateImportButton(); }
});

$("#wrong-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = event.currentTarget.querySelector("button"); setBusy(button, true, "正在理解题目并诊断，请稍候…");
  const form = new FormData(event.currentTarget);
  try {
    const result = await api("/api/wrong-problems", {method:"POST", body:JSON.stringify({problem_text:form.get("problem_text").trim(), student_answer:optional(form.get("student_answer")), correct_answer:optional(form.get("correct_answer")), student_solution:optional(form.get("student_solution"))})});
    $("#wrong-result").className = "analysis-grid";
    $("#wrong-result").innerHTML = `<article class="result-card"><h3>题目分析</h3><dl><dt>知识点</dt><dd>${tags(result.knowledge_points)}</dd><dt>题型</dt><dd>${escapeHTML(result.question_type)}</dd><dt>难度</dt><dd>${result.difficulty} / 5</dd><dt>题目摘要</dt><dd>${escapeHTML(result.problem_summary)}</dd></dl></article><article class="result-card"><h3>错误诊断</h3><dl><dt>错误类型</dt><dd><span class="badge neutral">${escapeHTML(result.error_type)}</span></dd><dt>原因</dt><dd>${escapeHTML(result.error_reason)}</dd><dt>相关知识点</dt><dd>${tags(result.related_knowledge_points)}</dd><dt>置信度</dt><dd>${Math.round(result.error_confidence*100)}%</dd></dl></article>`;
    toast(`错题 #${result.saved_problem_id} 已保存`); await loadDashboard(); await loadHistory();
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); }
});

$("#plan-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = event.currentTarget.querySelector("button"); setBusy(button, true);
  const total = Number(new FormData(event.currentTarget).get("total_questions"));
  try {
    const plan = await api("/api/training-plan", {method:"POST",body:JSON.stringify({total_questions:total})});
    $("#plan-result").innerHTML = `<article class="card"><h3>整体规划</h3><p>${escapeHTML(plan.plan_reason)}</p></article>` + plan.focus_items.map((focus,index) => `<article class="card focus-card"><span class="muted">训练目标 ${index+1}</span><h3>${escapeHTML(focus.knowledge_point)}</h3><div class="focus-count">${focus.question_count}<small> 题</small></div><p>目标难度：${focus.target_difficulty}</p><p>重点错误类型：${focus.focus_error_types.length ? tags(focus.focus_error_types) : "无可靠错因标签"}</p><p class="muted">${escapeHTML(focus.reason)}</p><button class="primary generate-focus" data-focus='${escapeHTML(JSON.stringify(focus))}' data-idle="基于此目标生成训练题">基于此目标生成训练题</button></article>`).join("");
    document.querySelectorAll(".generate-focus").forEach(button => button.addEventListener("click", () => generateQuestion(button)));
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); }
});

async function generateQuestion(button) {
  const focus = JSON.parse(button.dataset.focus); setBusy(button, true, "正在生成并验证题目…");
  try {
    const result = await api("/api/questions/generate", {method:"POST",body:JSON.stringify({knowledge_point:focus.knowledge_point,target_difficulty:focus.target_difficulty,focus_error_types:focus.focus_error_types})});
    if (result.generation_status !== "accepted") throw new Error("题目在最大验证次数内未通过，请稍后重试");
    currentQuestionId = result.question_id; $("#training-empty").classList.add("hidden"); $("#question-card").classList.remove("hidden");
    $("#question-id").textContent = `题目 #${result.question_id}`; $("#question-kp").innerHTML = tags(result.knowledge_points); $("#question-difficulty").textContent = `难度 ${result.difficulty}`; $("#question-text").textContent = result.question_text; $("#feedback-result").innerHTML = ""; $("#answer-form").reset(); location.hash = "training"; toast("训练题已生成并通过质量验证"); await loadHistory();
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); }
}

$("#answer-form").addEventListener("submit", async event => {
  event.preventDefault(); if (!currentQuestionId) return; const button = event.currentTarget.querySelector("button"); setBusy(button, true, "正在结合数学工具进行评价…");
  const form = new FormData(event.currentTarget);
  try {
    const result = await api("/api/training-feedback", {method:"POST",body:JSON.stringify({question_id:currentQuestionId,student_answer:form.get("student_answer").trim(),student_solution:optional(form.get("student_solution"))})});
    const sympyLabel = result.sympy_status === "unsupported" ? "数学工具未能自动判断" : result.sympy_status;
    const sympyClass = result.sympy_status === "equivalent" ? "success" : result.sympy_status === "not_equivalent" ? "danger" : "warning";
    $("#feedback-result").innerHTML = `<div class="feedback"><span class="badge ${result.is_correct ? "success" : "warning"}">${result.is_correct ? "正确" : "需要加强"}</span><h3>AI 评价</h3><p><strong>分析：</strong>${escapeHTML(result.error_reason)}</p><p>${escapeHTML(result.feedback)}</p><p>相关知识点：${tags(result.related_knowledge_points)}</p><p>置信度：${Math.round(result.confidence*100)}%</p><p><span class="badge ${sympyClass}">${escapeHTML(sympyLabel)}</span> <span class="muted">${escapeHTML(result.sympy_reason)}</span></p><div class="answer-box"><strong>标准答案</strong><p>${escapeHTML(result.reference_answer)}</p><strong>标准解析</strong><p>${escapeHTML(result.reference_solution)}</p></div></div>`;
    toast("本次训练反馈已保存"); await loadDashboard(); await loadHistory();
  } catch (error) { toast(`请求失败：${error.message}`, true); }
  finally { setBusy(button, false); }
});

async function loadHistory() {
  try {
    const [wrong, questions, attempts] = await Promise.all([api("/api/history/wrong-problems?limit=10"),api("/api/history/generated-questions?limit=10"),api("/api/history/training-attempts?limit=10")]);
    $("#wrong-history").innerHTML = wrong.length ? wrong.map(x => `<div class="history-item"><p>${tags(x.knowledge_points)}</p><small>#${x.id} · 难度 ${x.difficulty} · ${dateText(x.created_at)}</small></div>`).join("") : `<div class="empty">暂无错题</div>`;
    $("#question-history").innerHTML = questions.length ? questions.map(x => `<div class="history-item"><p>${escapeHTML(x.question_text)}</p><small>#${x.id} · ${tags(x.knowledge_points)} · ${dateText(x.created_at)}</small></div>`).join("") : `<div class="empty">暂无生成题</div>`;
    $("#attempt-history").innerHTML = attemptItems(attempts);
  } catch (error) { toast(`请求失败：${error.message}`, true); }
}
async function loadEvaluation() {
  try {
    const data = await api("/api/evaluation/latest");
    if (!data.available) { $("#evaluation-content").innerHTML = `<div class="empty">暂无真实 Pilot Evaluation 结果</div>`; return; }
    const metrics = [["Solver Symbolic Coverage",data.solver.symbolic_coverage,"ratio"],["Planner Weak Cluster Allocation",data.planner.weak_cluster_allocation_ratio,"ratio"],["Generator Accept Rate",data.generator.accept_rate,"ratio"],["Challenge Rejection Rate",data.verifier_challenge.corrupted_answer_rejection_rate,"ratio"],["LLM Calls",data.observability.total_llm_calls,"count"]];
    $("#evaluation-content").innerHTML = metrics.map(([label,value,type]) => `<article class="metric-card"><span>${label}</span><strong>${type === "ratio" ? `${(value*100).toFixed(1)}%` : value}</strong></article>`).join("") + `<article class="metric-card"><span>Dataset</span><strong>${data.case_count} cases</strong><small>${escapeHTML(data.dataset_name)}</small></article><p class="muted">${escapeHTML(data.limitation)} · Run ${escapeHTML(data.run_id)}</p>`;
  } catch (error) { $("#evaluation-content").innerHTML = `<p class="error">请求失败：${escapeHTML(error.message)}</p>`; }
}

$("#refresh-dashboard").addEventListener("click", loadDashboard); $("#refresh-profile").addEventListener("click", loadProfile); $("#refresh-history").addEventListener("click", loadHistory);
document.querySelectorAll("nav a").forEach(link => link.addEventListener("click", () => { document.querySelectorAll("nav a").forEach(x => x.classList.remove("active")); link.classList.add("active"); }));
loadHealth(); loadDashboard(); loadHistory(); loadEvaluation();
