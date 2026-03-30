import { useEffect, useState } from "react";
import { colors } from "./lib/tokens";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { MetricCard } from "./components/MetricCard";
import { Card } from "./components/Card";
import { DataTable } from "./components/DataTable";

interface SystemStatus {
  mode: string;
  total_trades: number;
  total_signals: number;
  total_evolution_events: number;
  kill_switch_status: string;
  approval_queue_size: number;
  evolution_paused: boolean;
}

interface Portfolio {
  total_value: number;
  cash: number;
  drawdown: number;
  sharpe_ratio: number | null;
}

interface Costs {
  total_cost_usd: number;
  cost_by_component: Record<string, number>;
}

function useFetch<T>(url: string, fallback: T): T {
  const [data, setData] = useState<T>(fallback);
  useEffect(() => {
    fetch(url).then((r) => r.json()).then(setData).catch(() => {});
  }, [url]);
  return data;
}

export function App() {
  const status = useFetch<SystemStatus>("/api/status", {
    mode: "loading...", total_trades: 0, total_signals: 0,
    total_evolution_events: 0, kill_switch_status: "unknown",
    approval_queue_size: 0, evolution_paused: false,
  });

  const portfolio = useFetch<Portfolio>("/api/portfolio/latest", {
    total_value: 0, cash: 0, drawdown: 0, sharpe_ratio: null,
  });

  const trades = useFetch<Record<string, unknown>[]>("/api/trades?limit=10", []);
  const signals = useFetch<Record<string, unknown>[]>("/api/signals?limit=10", []);
  const evolution = useFetch<Record<string, unknown>[]>("/api/evolution?limit=10", []);
  const costs = useFetch<Costs>("/api/costs", { total_cost_usd: 0, cost_by_component: {} });

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: colors.bg }}>
      <Sidebar />

      <div style={{ marginLeft: 240, flex: 1 }}>
        <TopBar
          mode={status.mode}
          totalTrades={status.total_trades}
          totalSignals={status.total_signals}
          totalEvolution={status.total_evolution_events}
        />

        <div style={{ padding: 24, maxWidth: 1440 }}>
          {/* Metric cards row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 24 }}>
            <MetricCard
              label="Portfolio Value"
              value={`$${portfolio.total_value.toLocaleString()}`}
            />
            <MetricCard
              label="Cash"
              value={`$${portfolio.cash.toLocaleString()}`}
            />
            <MetricCard
              label="Drawdown"
              value={`${(portfolio.drawdown * 100).toFixed(1)}%`}
              trendColor={portfolio.drawdown > 0.15 ? colors.red : portfolio.drawdown > 0.10 ? colors.amber : colors.green}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={portfolio.sharpe_ratio?.toFixed(2) ?? "—"}
            />
            <MetricCard
              label="LLM Spend"
              value={`$${costs.total_cost_usd.toFixed(3)}`}
            />
            <MetricCard
              label="Evolution Events"
              value={String(status.total_evolution_events)}
            />
          </div>

          {/* Operator status bar */}
          <div
            style={{
              display: "flex",
              gap: 24,
              padding: "12px 20px",
              background: colors.surfaceAlt,
              border: `1px solid ${colors.border}`,
              borderRadius: 10,
              marginBottom: 24,
              fontSize: 13,
            }}
          >
            <div>
              <span style={{ color: colors.textMuted, marginRight: 8 }}>Kill Switch:</span>
              <span style={{ color: colors.amber, fontWeight: 500 }}>{status.kill_switch_status}</span>
            </div>
            <div>
              <span style={{ color: colors.textMuted, marginRight: 8 }}>Approvals:</span>
              <span style={{ fontWeight: 500 }}>{status.approval_queue_size}</span>
            </div>
            <div>
              <span style={{ color: colors.textMuted, marginRight: 8 }}>Evolution:</span>
              <span style={{ color: status.evolution_paused ? colors.red : colors.green, fontWeight: 500 }}>
                {status.evolution_paused ? "Paused" : "Active"}
              </span>
            </div>
          </div>

          {/* Two-column layout */}
          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 24 }}>
            {/* Left: trades + signals */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <Card title="Recent Trades">
                <DataTable
                  columns={[
                    { key: "ticker", label: "Ticker", mono: true },
                    { key: "strategy_skill", label: "Strategy" },
                    {
                      key: "direction", label: "Dir",
                      render: (v) => (
                        <span style={{ color: v === "BUY" ? colors.green : colors.red, fontWeight: 500 }}>
                          {String(v)}
                        </span>
                      ),
                    },
                    {
                      key: "pnl", label: "PnL", align: "right", mono: true,
                      render: (v) => {
                        const n = Number(v);
                        return (
                          <span style={{ color: n >= 0 ? colors.green : colors.red }}>
                            {v != null ? `$${n.toFixed(0)}` : "—"}
                          </span>
                        );
                      },
                    },
                  ]}
                  rows={trades}
                  emptyMessage="No trades yet"
                />
              </Card>

              <Card title="Recent Signals">
                <DataTable
                  columns={[
                    { key: "source", label: "Source", mono: true },
                    { key: "source_entity", label: "Entity" },
                    { key: "signal_type", label: "Type" },
                    {
                      key: "confidence", label: "Confidence", align: "right", mono: true,
                      render: (v) => (
                        <span style={{ color: colors.blue, fontWeight: 500 }}>
                          {`${(Number(v) * 100).toFixed(0)}%`}
                        </span>
                      ),
                    },
                  ]}
                  rows={signals}
                  emptyMessage="No signals yet"
                />
              </Card>
            </div>

            {/* Right: costs + evolution */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <Card title="LLM Cost Breakdown">
                {Object.entries(costs.cost_by_component).length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {Object.entries(costs.cost_by_component).map(([comp, cost]) => (
                      <div key={comp} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                        <span style={{ color: colors.textSecondary }}>{comp}</span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 500 }}>
                          ${cost.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: colors.textMuted, fontSize: 13 }}>No LLM costs recorded</div>
                )}
              </Card>

              <Card title="Evolution Events">
                <DataTable
                  columns={[
                    {
                      key: "event_type", label: "Type",
                      render: (v) => {
                        const bg = v === "FIX" ? colors.amberSoft : v === "DERIVED" ? colors.blueSoft : colors.accentSoft;
                        const fg = v === "FIX" ? colors.amber : v === "DERIVED" ? colors.blue : colors.accent;
                        return (
                          <span style={{
                            background: bg, color: fg, padding: "2px 6px",
                            borderRadius: 4, fontSize: 11, fontWeight: 600,
                            fontFamily: "'JetBrains Mono', monospace",
                          }}>
                            {String(v)}
                          </span>
                        );
                      },
                    },
                    { key: "child_skill", label: "Skill", mono: true },
                    { key: "trigger_reason", label: "Reason" },
                  ]}
                  rows={evolution}
                  emptyMessage="No evolution events yet"
                />
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
