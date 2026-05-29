import type { TimelineEvent } from "../sim/types";

export function Timeline({ events, stepMs }: { events: TimelineEvent[]; stepMs: number }) {
  const lanes = Array.from(new Set(events.map((event) => event.lane)));
  const safeStep = Math.max(stepMs, 1);

  return (
    <section className="panel timeline-panel">
      <div className="section-heading">
        <h2>简化训练时间线</h2>
        <span>{Math.round(stepMs).toLocaleString()} ms / step</span>
      </div>
      <div className="timeline">
        {lanes.map((lane) => (
          <div className="timeline-lane" key={lane}>
            <span className="lane-label">{lane}</span>
            <div className="lane-track">
              {events
                .filter((event) => event.lane === lane)
                .map((event) => (
                  <div
                    className={`timeline-event ${event.kind}`}
                    key={`${event.label}-${event.startMs}`}
                    title={`${event.label}: ${event.durationMs} ms`}
                    style={{
                      left: `${(event.startMs / safeStep) * 100}%`,
                      width: `${Math.max(2, (event.durationMs / safeStep) * 100)}%`,
                    }}
                  >
                    {event.label}
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
