document.addEventListener("DOMContentLoaded", async () => {
  const canvasActivity = document.getElementById("chartActivity");
  if (!canvasActivity) return;

  const charts = {};

  function themeColors() {
    const dark = document.documentElement.getAttribute("data-bs-theme") === "dark";
    return {
      text: dark ? "#e2e8f0" : "#334155",
      grid: dark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.06)",
    };
  }

  function destroy(name) {
    if (charts[name]) { charts[name].destroy(); delete charts[name]; }
  }

  async function load() {
    let data;
    try {
      data = await window.apiFetch("/api/stats/", { method: "GET" });
    } catch (err) {
      console.error("Failed to load stats:", err);
      return;
    }

    const colors = themeColors();

    document.getElementById("statTotal").textContent = data.total_predictions ?? 0;
    document.getElementById("statMost").textContent =
      data.most_detected ? data.most_detected.activity.replace(/_/g, " ") : "—";
    document.getElementById("statConfidence").textContent =
      `${Math.round((data.avg_confidence || 0) * 100)}%`;
    document.getElementById("statTime").textContent =
      `${Math.round(data.avg_processing_time || 0)} ms`;

    // ---- Activity frequency (bar) --------------------------------------
    destroy("activity");
    charts.activity = new Chart(canvasActivity, {
      type: "bar",
      data: {
        labels: (data.by_activity || []).map(r => r.activity.replace(/_/g, " ")),
        datasets: [{
          label: "Predictions",
          data: (data.by_activity || []).map(r => r.count),
          backgroundColor: "rgba(79, 70, 229, 0.7)",
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: colors.text }, grid: { display: false } },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid }, beginAtZero: true },
        },
      },
    });

    // ---- Predictions over time (line) ----------------------------------
    destroy("time");
    charts.time = new Chart(document.getElementById("chartTime"), {
      type: "line",
      data: {
        labels: (data.over_time || []).map(r => r.date),
        datasets: [{
          label: "Predictions",
          data: (data.over_time || []).map(r => r.count),
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.15)",
          fill: true,
          tension: 0.35,
          pointRadius: 3,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: colors.text }, grid: { display: false } },
          y: { ticks: { color: colors.text }, grid: { color: colors.grid }, beginAtZero: true },
        },
      },
    });

    // ---- Confidence over time (line) -----------------------------------
    destroy("confidence");
    charts.confidence = new Chart(document.getElementById("chartConfidence"), {
      type: "line",
      data: {
        labels: (data.over_time || []).map(r => r.date),
        datasets: [{
          label: "Average confidence",
          data: (data.over_time || []).map(r => Math.round((r.avg_confidence || 0) * 100)),
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          fill: true,
          tension: 0.35,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: colors.text }, grid: { display: false } },
          y: {
            min: 0, max: 100,
            ticks: { color: colors.text, callback: v => `${v}%` },
            grid: { color: colors.grid },
          },
        },
      },
    });
  }

  document.getElementById("btnRefresh").addEventListener("click", load);
  await load();
});