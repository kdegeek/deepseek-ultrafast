#include "stream.h"
#include <cstdlib>
#include <iostream>

static void check(bool condition) {
    if (!condition) { std::cerr << "stream contract failed\n"; std::exit(1); }
}
int main() {
    text_stream a({"STOP", "END"});
    check(a.push("hello ST") == "hello ");
    check(a.push("O") == "");
    check(a.push("P hidden") == "" && a.stopped);
    check(a.finish().empty());
    text_stream b({"STOP"});
    check(b.push("ST") == "");
    check(b.push("X") == "STX");
    check(b.push("ST") == "");
    check(b.finish() == "ST");
    text_stream c({});
    check(c.push("\xE4") == "");
    check(c.push("\xBD") == "");
    check(c.push("\xA0!") == "\xE4\xBD\xA0!");
    check(c.finish().empty());
    text_stream d({"\xE4\xBD\xA0"});
    check(d.push("prefix \xE4") == "prefix ");
    check(d.push("\xBD\xA0 suffix").empty() && d.stopped);
    text_stream e({"abcd", "bc"});
    check(e.push("abc rest") == "a" && e.stopped);
    text_stream f({});
    check(f.push("\xF0\x9F").empty());
    check(f.finish() == "\xEF\xBF\xBD");
    text_stream g({});
    check(g.push("\xC2").empty());
    check(g.push("\xE2\x82\xAC") == "\xEF\xBF\xBD\xE2\x82\xAC");
    check(g.finish().empty());
    std::cout << "stop boundaries, overlap, UTF-8 and truncation passed\n";
}
