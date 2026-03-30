import { colors, fonts } from "../lib/tokens";

interface MetricCardProps {
  label: string;
  value: string;
  trend?: string;
  trendColor?: string;
}

export function MetricCard({ label, value, trend, trendColor }: MetricCardProps) {
  return (
    <div
      style={{
        background: colors.surface,
        border: `1px solid ${colors.border}`,
        borderRadius: 10,
        padding: "16px 20px",
      }}
    >
      <div style={{ fontSize: 11, color: colors.textMuted, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.03em" }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 600, fontFamily: fonts.mono, color: colors.text }}>
        {value}
      </div>
      {trend && (
        <div style={{ fontSize: 11, color: trendColor || colors.textMuted, marginTop: 4 }}>
          {trend}
        </div>
      )}
    </div>
  );
}
