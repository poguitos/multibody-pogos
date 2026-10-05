# How to stop at, or act on, an event

<!-- example: tests/examples/ex_phase3.cpp#events -->

```cpp
Event impact;
impact.name = "impact";
impact.function = [](const Simulator& s) { return s.q(0) - 0.1; };   // height above contact
impact.direction = -1;                                                // falling through zero only
impact.action = [](Simulator& s) { s.v(0) = -0.8 * s.v(0); };         // rebound, restitution 0.8
sim.events.push_back(impact);
sim.run(2.0, 0.01);
for (const EventRecord& e : sim.event_log()) {
    out << sim.events[e.event].name << " at " << e.time << " s\n";   // to 1e-10 s
}
```

An event is a function of the simulator's state whose sign change marks
something: an impact, a limit, a switch. When a step's two ends give the
function opposite signs, the step is integrated again to the crossing,
located to `event_tolerance`, the action runs there (just past the crossing:
the function already has its new sign), and the rest of the step follows.
Give a direction (`-1`, falling) when only one way matters; set `stop` to end
the run at the event. Two crossings of one function within one step are not
seen, so keep the step shorter than the time between them; events that
follow each other a tolerance apart are chattering (MBD-K091), the sign of a
law that should be regularised.

More: [the kernel, events](../../kernel.md#forces-and-simulation).
