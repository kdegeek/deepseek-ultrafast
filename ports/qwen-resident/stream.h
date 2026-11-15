#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

// Preserve stops split over tokens and UTF-8 split over tokenizer byte pieces.
// The withheld suffix is at most the longest stop plus one UTF-8 codepoint.
class text_stream {
    std::vector<std::string> stops;
    std::string pending;
public:
    bool stopped = false;
    explicit text_stream(std::vector<std::string> values) : stops(std::move(values)) {}

    // Positive = complete character, zero = incomplete, -1 = malformed byte.
    static int character(const std::string & value, size_t i, size_t limit) {
        const auto c = static_cast<unsigned char>(value[i]);
        if (c < 0x80) return 1;
        const int n = c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : -1;
        if (n < 0) return -1;
        for (int j = 1; j < n; ++j) {
            if (i + j >= limit) return 0;
            const auto next = static_cast<unsigned char>(value[i + j]);
            if (next < 0x80 || next > 0xBF) return -1;
            if (j == 1 && ((c == 0xE0 && next < 0xA0) || (c == 0xED && next >= 0xA0) ||
                           (c == 0xF0 && next < 0x90) || (c == 0xF4 && next >= 0x90))) return -1;
        }
        return n;
    }
    static size_t complete_utf8(const std::string & value, size_t limit) {
        size_t i = 0;
        while (i < limit) {
            const int n = character(value, i, limit);
            if (n == 0) break;
            i += n < 0 ? 1 : n;
        }
        return i;
    }
    static std::string sanitize(const std::string & value) {
        std::string result;
        for (size_t i = 0; i < value.size();) {
            const int n = character(value, i, value.size());
            if (n <= 0) { result += "\xEF\xBF\xBD"; ++i; }
            else { result.append(value, i, n); i += n; }
        }
        return result;
    }

    std::string push(const std::string & piece) {
        if (stopped) return {};
        pending += piece;
        size_t cut = pending.size();
        for (const auto & stop : stops) {
            const auto pos = pending.find(stop);
            if (pos != std::string::npos) {
                cut = std::min(cut, pos);
                stopped = true;
            }
        }
        if (!stopped) {
            size_t hold = 0;
            for (const auto & stop : stops) {
                for (size_t n = 1; n < stop.size() && n <= pending.size(); ++n) {
                    if (pending.compare(pending.size() - n, n, stop, 0, n) == 0) hold = std::max(hold, n);
                }
            }
            cut -= hold;
        }
        cut = complete_utf8(pending, cut);
        std::string result = sanitize(pending.substr(0, cut));
        if (stopped) pending.clear();
        else pending.erase(0, cut);
        return result;
    }

    std::string finish() {
        std::string result;
        if (!stopped) {
            const auto n = complete_utf8(pending, pending.size());
            result = sanitize(pending.substr(0, n));
            if (n < pending.size()) result += "\xEF\xBF\xBD";
        }
        pending.clear();
        return result;
    }
};
