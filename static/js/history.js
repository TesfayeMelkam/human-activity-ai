document.addEventListener("DOMContentLoaded", () => {
  const tbody = document.getElementById("historyBody");
  if (!tbody) return;

  const info = document.getElementById("historyInfo");
  const pagination = document.getElementById("historyPagination");
  const search = document.getElementById("filterSearch");
  const source = document.getElementById("filterSource");
  const pageSize = document.getElementById("filterPageSize");
  const btnApply = document.getElementById("btnApplyFilter");
  const btnClear = document.getElementById("btnClearHistory");

  let state = { page: 1, pageSize: 10, search: "", source: "" };

  async function load() {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">Loading…</td></tr>';

    const params = new URLSearchParams({
      page: state.page,
      page_size: state.pageSize,
    });
    if (state.search) params.set("search", state.search);
    if (state.source) params.set("source", state.source);

    let data;
    try {
      data = await window.apiFetch(`/api/history/?${params.toString()}`, { method: "GET" });
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-danger text-center py-4">${err.message}</td></tr>`;
      return;
    }

    if (!data.results.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">No predictions found.</td></tr>';
    } else {
      tbody.innerHTML = data.results.map(row => `
        <tr>
          <td>${row.image_url
              ? `<img src="${row.image_url}" alt="" style="width:48px;height:48px;object-fit:cover;border-radius:8px;">`
              : '<span class="text-muted">—</span>'}</td>
          <td><span class="badge ${row.is_uncertain ? 'bg-warning text-dark' : 'bg-primary'}">${row.activity.replace(/_/g, " ")}</span></td>
          <td style="min-width:140px;">
            <div class="progress" style="height:8px;">
              <div class="progress-bar ${row.is_uncertain ? 'bg-warning' : 'bg-success'}"
                   style="width:${row.confidence_percent}%"></div>
            </div>
            <small class="text-muted">${row.confidence_percent}%</small>
          </td>
          <td>${Math.round(row.processing_time)} ms</td>
          <td>${row.source}</td>
          <td><small>${new Date(row.created_at).toLocaleString()}</small></td>
          <td>
            <button class="btn btn-sm btn-outline-danger" data-delete="${row.id}">
              <i class="bi bi-trash"></i>
            </button>
          </td>
        </tr>`).join("");
    }

    info.textContent = `${data.count} record(s) — page ${data.page} of ${data.total_pages}`;
    renderPagination(data.page, data.total_pages);

    tbody.querySelectorAll("[data-delete]").forEach(btn => {
      btn.addEventListener("click", async () => {
        if (!confirm("Delete this prediction?")) return;
        try {
          await window.apiFetch(`/api/history/?id=${btn.dataset.delete}`, { method: "DELETE" });
          load();
        } catch (err) { alert(err.message); }
      });
    });
  }

  function renderPagination(current, total) {
    pagination.innerHTML = "";
    const add = (label, page, disabled, active) => {
      const li = document.createElement("li");
      li.className = `page-item ${disabled ? "disabled" : ""} ${active ? "active" : ""}`;
      li.innerHTML = `<a class="page-link" href="#">${label}</a>`;
      li.addEventListener("click", e => {
        e.preventDefault();
        if (disabled || active) return;
        state.page = page;
        load();
      });
      pagination.appendChild(li);
    };

    add("«", Math.max(current - 1, 1), current === 1, false);
    const start = Math.max(1, current - 2);
    const end = Math.min(total, start + 4);
    for (let p = start; p <= end; p++) add(p, p, false, p === current);
    add("»", Math.min(current + 1, total), current === total, false);
  }

  btnApply.addEventListener("click", () => {
    state.page = 1;
    state.search = search.value.trim();
    state.source = source.value;
    state.pageSize = parseInt(pageSize.value, 10);
    load();
  });

  search.addEventListener("keydown", e => {
    if (e.key === "Enter") btnApply.click();
  });

  btnClear.addEventListener("click", async () => {
    if (!confirm("Delete ALL prediction history? This cannot be undone.")) return;
    try {
      await window.apiFetch("/api/history/?all=true", { method: "DELETE" });
      load();
    } catch (err) { alert(err.message); }
  });

  load();
});