#pragma once

// A small SVG line plot for the integrators' verification report (plan task
// 4.9): logarithmic or linear axes, one line with markers per series, guide
// lines of a given slope, a legend. Enough for error-against-step and
// error-against-cost figures without a plotting library; the files are
// plain text, so a figure's change shows in a diff.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace mbd_plot {

struct Series {
    std::string name;
    std::vector<double> x, y;
    bool dashed{false};
    bool markers{true};   // off for a dense time series
};

/// A guide line of slope p (on log-log axes: y proportional to x^p), from x0
/// to x1, through (x0, y0).
struct Slope {
    double p, x0, y0, x1;
    std::string label;
};

class Plot {
public:
    std::string title, xlabel, ylabel;
    bool logx{true}, logy{true};
    std::vector<Series> series;
    std::vector<Slope> slopes;

    void write(const std::string& path) const
    {
        // Data range over every finite point that the axes can show.
        double x_lo = HUGE_VAL, x_hi = -HUGE_VAL, y_lo = HUGE_VAL, y_hi = -HUGE_VAL;
        for (const Series& s : series) {
            for (std::size_t i = 0; i < s.x.size(); ++i) {
                if (!shown(s.x[i], s.y[i])) continue;
                x_lo = std::min(x_lo, s.x[i]);
                x_hi = std::max(x_hi, s.x[i]);
                y_lo = std::min(y_lo, s.y[i]);
                y_hi = std::max(y_hi, s.y[i]);
            }
        }
        if (!(x_lo <= x_hi)) x_lo = 1.0, x_hi = 10.0;
        if (!(y_lo <= y_hi)) y_lo = 1.0, y_hi = 10.0;
        const Axis ax = make_axis(x_lo, x_hi, logx);
        const Axis ay = make_axis(y_lo, y_hi, logy);

        std::ostringstream o;
        o << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << kWidth << "\" height=\"" << kHeight
          << "\" viewBox=\"0 0 " << kWidth << ' ' << kHeight << "\" font-family=\"Helvetica, Arial, sans-serif\""
          << " font-size=\"12\">\n";
        o << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
        o << "<text x=\"" << (kLeft + kPlotW / 2) << "\" y=\"22\" text-anchor=\"middle\" font-size=\"14\">"
          << escape(title) << "</text>\n";

        // Grid and ticks.
        for (const Tick& t : ax.ticks) {
            const double X = px(ax, t.value);
            o << line(X, kTop, X, kTop + kPlotH, t.major ? "#d8d8d8" : "#efefef");
            if (t.labelled) {
                o << "<text x=\"" << X << "\" y=\"" << (kTop + kPlotH + 16) << "\" text-anchor=\"middle\">"
                  << label(t.value, logx) << "</text>\n";
            }
        }
        for (const Tick& t : ay.ticks) {
            const double Y = py(ay, t.value);
            o << line(kLeft, Y, kLeft + kPlotW, Y, t.major ? "#d8d8d8" : "#efefef");
            if (t.labelled) {
                o << "<text x=\"" << (kLeft - 6) << "\" y=\"" << (Y + 4) << "\" text-anchor=\"end\">"
                  << label(t.value, logy) << "</text>\n";
            }
        }
        o << "<rect x=\"" << kLeft << "\" y=\"" << kTop << "\" width=\"" << kPlotW << "\" height=\"" << kPlotH
          << "\" fill=\"none\" stroke=\"#333333\"/>\n";
        o << "<text x=\"" << (kLeft + kPlotW / 2) << "\" y=\"" << (kHeight - 12)
          << "\" text-anchor=\"middle\">" << escape(xlabel) << "</text>\n";
        o << "<text transform=\"translate(16," << (kTop + kPlotH / 2) << ") rotate(-90)\" text-anchor=\"middle\">"
          << escape(ylabel) << "</text>\n";

        // Guide slopes, clipped to the plot.
        o << "<clipPath id=\"plot\"><rect x=\"" << kLeft << "\" y=\"" << kTop << "\" width=\"" << kPlotW
          << "\" height=\"" << kPlotH << "\"/></clipPath>\n<g clip-path=\"url(#plot)\">\n";
        for (const Slope& s : slopes) {
            const double y1 = s.y0 * std::pow(s.x1 / s.x0, s.p);
            o << "<line x1=\"" << px(ax, s.x0) << "\" y1=\"" << py(ay, s.y0) << "\" x2=\"" << px(ax, s.x1)
              << "\" y2=\"" << py(ay, y1) << "\" stroke=\"#888888\" stroke-dasharray=\"2,3\"/>\n";
            // The label at the guide's left end, right-aligned.
            const bool x0_left = s.x0 < s.x1;
            o << "<text x=\"" << px(ax, x0_left ? s.x0 : s.x1) - 4 << "\" y=\"" << py(ay, x0_left ? s.y0 : y1) + 4
              << "\" text-anchor=\"end\" fill=\"#666666\" font-size=\"11\">" << escape(s.label) << "</text>\n";
        }

        // Series: lines through the points shown, then the markers.
        for (std::size_t k = 0; k < series.size(); ++k) {
            const Series& s = series[k];
            const char* colour = kColours[k % kColourCount];
            std::ostringstream pts;
            int n = 0;
            for (std::size_t i = 0; i < s.x.size(); ++i) {
                if (!shown(s.x[i], s.y[i])) continue;
                pts << px(ax, s.x[i]) << ',' << py(ay, s.y[i]) << ' ';
                ++n;
            }
            if (n > 1) {
                o << "<polyline fill=\"none\" stroke=\"" << colour << "\" stroke-width=\"1.6\""
                  << (s.dashed ? " stroke-dasharray=\"6,3\"" : "") << " points=\"" << pts.str() << "\"/>\n";
            }
            for (std::size_t i = 0; i < s.x.size(); ++i) {
                if (s.markers && shown(s.x[i], s.y[i])) o << marker(k, px(ax, s.x[i]), py(ay, s.y[i]), colour);
            }
        }
        o << "</g>\n";

        // Legend, right of the plot.
        for (std::size_t k = 0; k < series.size(); ++k) {
            const double Y = kTop + 10 + 18.0 * static_cast<double>(k);
            const double X = kLeft + kPlotW + 14;
            const char* colour = kColours[k % kColourCount];
            o << "<line x1=\"" << X << "\" y1=\"" << Y << "\" x2=\"" << X + 24 << "\" y2=\"" << Y << "\" stroke=\""
              << colour << "\" stroke-width=\"1.6\"" << (series[k].dashed ? " stroke-dasharray=\"6,3\"" : "")
              << "/>\n";
            if (series[k].markers) o << marker(k, X + 12, Y, colour);
            o << "<text x=\"" << X + 30 << "\" y=\"" << Y + 4 << "\">" << escape(series[k].name) << "</text>\n";
        }
        o << "</svg>\n";

        std::ofstream f(path, std::ios::binary);
        f << o.str();
    }

private:
    static constexpr double kWidth = 760, kHeight = 440;
    static constexpr double kLeft = 72, kTop = 36, kPlotW = 460, kPlotH = 340;
    // Okabe and Ito's palette, distinguishable with the common colour
    // deficiencies; yellow left out (too pale on white).
    static constexpr int kColourCount = 8;
    static constexpr const char* kColours[kColourCount] = {"#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                                           "#E69F00", "#56B4E9", "#000000", "#999999"};

    struct Tick {
        double value;
        bool major;
        bool labelled;
    };
    struct Axis {
        double lo, hi;   // in axis units (log10 for a log axis)
        bool log;
        std::vector<Tick> ticks;
    };

    bool shown(double x, double y) const
    {
        return std::isfinite(x) && std::isfinite(y) && (!logx || x > 0.0) && (!logy || y > 0.0);
    }

    static Axis make_axis(double lo, double hi, bool log)
    {
        Axis a;
        a.log = log;
        if (log) {
            // The data's range, padded a little, and at least a decade.
            double l = std::log10(lo), h = std::log10(hi);
            if (h - l < 1.0) {
                const double mid = 0.5 * (l + h);
                l = mid - 0.5;
                h = mid + 0.5;
            }
            const double pad = 0.04 * (h - l);
            a.lo = l - pad;
            a.hi = h + pad;
            const double span = a.hi - a.lo;
            const int every = span > 10.0 ? 2 : 1;   // label every other decade on a long axis
            for (int d = static_cast<int>(std::floor(a.lo)); d <= static_cast<int>(std::ceil(a.hi)); ++d) {
                for (int m = 1; m <= 9; ++m) {
                    if (m > 1 && span > 6.0) break;   // no minor ticks on a long axis
                    const double v = m * std::pow(10.0, d);
                    if (std::log10(v) < a.lo || std::log10(v) > a.hi) continue;
                    const bool major = m == 1;
                    const bool labelled = (major && d % every == 0) || (span < 2.0 && (m == 2 || m == 5));
                    a.ticks.push_back({v, major, labelled});
                }
            }
        } else {
            if (hi - lo <= 0.0) {
                lo -= 0.5;
                hi += 0.5;
            }
            const double raw = (hi - lo) / 5.0;
            const double mag = std::pow(10.0, std::floor(std::log10(raw)));
            const double r = raw / mag;
            const double step = (r < 1.5 ? 1.0 : r < 3.5 ? 2.0 : r < 7.5 ? 5.0 : 10.0) * mag;
            a.lo = std::floor(lo / step) * step;
            a.hi = std::ceil(hi / step) * step;
            for (double v = a.lo; v <= a.hi + 0.5 * step; v += step) {
                a.ticks.push_back({std::abs(v) < 1e-12 * step ? 0.0 : v, true, true});
            }
        }
        return a;
    }

    static double frac(const Axis& a, double v) { return ((a.log ? std::log10(v) : v) - a.lo) / (a.hi - a.lo); }
    static double px(const Axis& a, double v) { return kLeft + kPlotW * frac(a, v); }
    static double py(const Axis& a, double v) { return kTop + kPlotH * (1.0 - frac(a, v)); }

    static std::string line(double x1, double y1, double x2, double y2, const char* colour)
    {
        std::ostringstream o;
        o << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" stroke=\""
          << colour << "\"/>\n";
        return o.str();
    }

    /// On a log axis 10 with a raised exponent, or m x 10^e; a linear value
    /// plainly.
    static std::string label(double v, bool log)
    {
        std::ostringstream o;
        if (log) {
            const int e = static_cast<int>(std::floor(std::log10(v) + 1e-9));
            const int m = static_cast<int>(std::lround(v / std::pow(10.0, e)));
            if (e == 0) return std::to_string(m);
            if (e == 1 && m == 1) return "10";
            if (m != 1) o << m << "&#215;";
            o << "10<tspan dy=\"-6\" font-size=\"9\">" << e << "</tspan>";
        } else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%g", v);
            o << buf;
        }
        return o.str();
    }

    /// Markers cycle through shapes as well as colours.
    static std::string marker(std::size_t k, double x, double y, const char* colour)
    {
        std::ostringstream o;
        const double r = 3.5;
        switch (k % 5) {
            case 0:
                o << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << r << "\" fill=\"" << colour << "\"/>";
                break;
            case 1:
                o << "<rect x=\"" << x - r << "\" y=\"" << y - r << "\" width=\"" << 2 * r << "\" height=\"" << 2 * r
                  << "\" fill=\"" << colour << "\"/>";
                break;
            case 2:
                o << "<polygon points=\"" << x << ',' << y - r - 1 << ' ' << x - r - 0.5 << ',' << y + r << ' '
                  << x + r + 0.5 << ',' << y + r << "\" fill=\"" << colour << "\"/>";
                break;
            case 3:
                o << "<polygon points=\"" << x << ',' << y - r - 1 << ' ' << x + r + 1 << ',' << y << ' ' << x << ','
                  << y + r + 1 << ' ' << x - r - 1 << ',' << y << "\" fill=\"" << colour << "\"/>";
                break;
            default:
                o << "<circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"" << r << "\" fill=\"#ffffff\" stroke=\""
                  << colour << "\" stroke-width=\"1.5\"/>";
                break;
        }
        o << '\n';
        return o.str();
    }

    static std::string escape(const std::string& s)
    {
        std::string out;
        for (const char c : s) {
            if (c == '&') out += "&amp;";
            else if (c == '<') out += "&lt;";
            else if (c == '>') out += "&gt;";
            else out += c;
        }
        return out;
    }
};

} // namespace mbd_plot
