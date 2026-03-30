import { colors } from "../lib/tokens";

interface CardProps {
  title: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}

export function Card({ title, children, style }: CardProps) {
  return (
    <div
      style={{
        background: colors.surface,
        border: `1px solid ${colors.border}`,
        borderRadius: 10,
        padding: 20,
        ...style,
      }}
    >
      <h3
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: colors.text,
          margin: "0 0 16px",
        }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}
