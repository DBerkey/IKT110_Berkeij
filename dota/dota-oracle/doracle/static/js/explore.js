function formatHeroStats(data) {
    if (!data) {
        return `<p style="color:#8b949e;">No data available.</p>`;
    }
    if (data.error) {
        return `<p style="color:#f85149;">Error: ${data.error}</p>`;
    }

    const heroImage = `/static/img/avatar-sb/${data.hero_id}.png`;
    const winrateLine = data.win_rate_percent
        ? `${data.win_rate_percent}${data.games_sampled ? ` · ${Number(data.games_sampled).toLocaleString()} games` : ""}`
        : "Not enough data";

    const safeScore = data.safe_pick_score !== undefined && data.safe_pick_score !== null
        ? Number(data.safe_pick_score).toFixed(3)
        : "—";

    const counterStrength = data.avg_counter_strength !== undefined && data.avg_counter_strength !== null
         ? Number(data.avg_counter_strength).toFixed(3)
        : "—";

    let pairSection = `<div style="color:#8b949e;">Not available</div>`;
    if (Array.isArray(data.best_pairs) && data.best_pairs.length > 0) {
        const pairCards = data.best_pairs.map((pair, idx) => {
            const synergy = pair.synergy_score !== undefined && pair.synergy_score !== null
                ? Number(pair.synergy_score).toFixed(3)
                : "—";
            const imgSrc = pair.image || `/static/img/avatar-sb/${pair.id}.png`;
            return `
                <div style="position:relative;border:1px solid #1f2329;border-radius:8px;background:#161b22;overflow:hidden;">
                    <div style="position:absolute;top:0;left:0;width:30%;height:100%;background:url('${imgSrc}') center/cover no-repeat;"></div>
                    <div style="position:absolute;top:0;left:0;width:30%;height:100%;background:linear-gradient(90deg, rgba(22, 27, 34, 0) 0%, rgba(22, 27, 34,1) 95%);"></div>
                    <div style="position:absolute;top:0;left:30%;width:70%;height:100%;background:#161b22;"></div>
                    <div style="position:relative;z-index:1;display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem 0.5rem calc(0.75rem + 30%);">
                        <div>
                            <div style="font-weight:600;">${pair.name}</div>
                            <div style="font-size:0.8rem;color:#8b949e;">Synergy ${synergy}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join("\n");
        pairSection = `<div style="display:flex;flex-direction:column;gap:0.6rem;">${pairCards}</div>`;
    }

    return `
        <div style="font-family:'Segoe UI', Arial, sans-serif;background:#0d1117;color:#f0f6fc;padding:1rem;border-radius:10px;box-shadow:0 12px 30px rgba(0,0,0,0.35);">
            <div style="margin-bottom:0.8rem;position:relative;border-radius:8px;overflow:hidden;">
                <img src="${heroImage}" alt="${data.hero}" style="width:100%;height:140px;object-fit:cover;filter:brightness(0.6);">
                <div style="position:absolute;top:0;bottom:0;width:100%;height:100%;background:linear-gradient(180deg, rgba(13, 17, 23, 0) 0%, rgba(13, 17, 23,1) 95%);"></div>
                <div style="position:absolute;left:0;right:0;bottom:0;padding:0.85rem;">
                    <div style="font-size:0.85rem;text-transform:uppercase;letter-spacing:0.08em;color:#8b949e;">Hero Overview</div>
                    <div style="font-size:1.6rem;font-weight:600;">${data.hero}</div>
                </div>
            </div>
            <div style="display:flex;flex-direction:column;gap:0.75rem;">
                <div>
                    <div style="font-size:0.75rem;text-transform:uppercase;color:#8b949e;">Estimated Win Rate</div>
                    <div style="font-size:1.15rem;font-weight:600;color:#3fb950;">${winrateLine}</div>
                </div>
                <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                    <div style="flex:1;min-width:120px;">
                        <div style="font-size:0.75rem;text-transform:uppercase;color:#8b949e;">Safe Pick Score</div>
                        <div style="font-size:1rem;">${safeScore}</div>
                    </div>
                    <div style="flex:1;min-width:120px;">
                        <div style="font-size:0.75rem;text-transform:uppercase;color:#8b949e;">Avg Counter Strength</div>
                        <div style="font-size:1rem;">${counterStrength}</div>
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem;text-transform:uppercase;color:#8b949e;">Top Synergy Partners</div>
                    ${pairSection}
                </div>
            </div>
        </div>
    `;
}

$(document).ready(function () {
    const output = $("#outputExplore");

    $(".hero-image").click(function () {
        const hero_id = $(this).data("heroid");
        const url = `/stats/${hero_id}`;
        $.get(url, function (data) {
            const formatted = formatHeroStats(data);
            output.html(formatted);
        });
    });
});