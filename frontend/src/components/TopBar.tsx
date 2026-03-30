import { colors } from "../lib/tokens";

interface TopBarProps {
  mode: string;
  totalTrades: number;
  totalSignals: number;
  totalEvolution: number;
}

export function TopBar({ mode, totalTrades, totalSignals, totalEvolution }: TopBarProps) {
  return (
    <header
      style={{
        height: 52,
        background: colors.surface,
        borderBottom: `1px solid ${colors.border}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 24px",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <h1 style={{ fontSize: 16, fontWeight: 600, color: colors.text, margin: 0 }}>
          Dashboard
        </h1>
        <span
          style={{
            fontSize: 11,
            fontWeight: 500,
            color: colors.accentText,
            background: colors.accentSoft,
            padding: "2px 8px",
            borderRadius: 4,
          }}
        >
          {mode}
        </span>
      </div>
      <div style={{ display: "flex", gap: 20, fontSize: 12, color: colors.textMuted }}>
        <span>{totalTrades} trades</span>
        <span>{totalSignals} signals</span>
        <span>{totalEvolution} evolutions</span>
      </div>
    </header>
  );
}
