// liyab-tok — a model's tokenizer on its own: encodes a text, compares the
// ids with a reference, and measures encoding speed. Only the GGUF's metadata
// is read, so a vocabulary-only file (tools/gguf_vocab.py) works as well as
// the full model.
//
//     liyab-tok -m model.gguf -f text.txt                    # print the ids
//     liyab-tok -m model.gguf -f text.txt --ids ref.ids      # compare (exit 1 on a difference)
//     liyab-tok -m model.gguf -f text.txt --bench 50         # encoding speed
//     liyab-tok -m model.gguf --render request.json           # the model's chat template applied
//     liyab-tok -m model.gguf --format '{"enable_thinking": false}'   # its layout of turns
//
// tools/tokenizer_reference.py writes the reference ids with the model's own
// Hugging Face tokenizer. Both sides encode the text without BOS/EOS by
// default (the tokenizer, not the special-token rule, is under test); --bos
// adds a BOS by the model's rule (tokenizer.ggml.add_bos_token), to compare
// with a reference written with --special. On a difference the first
// differing position is printed with the pieces around it.
//
// --render renders a chat request (Engine::render_chat's JSON) with the
// GGUF's tokenizer.chat_template; --format prints Engine::chat_format's
// result for a probe (template variables, optional reply_prefix).
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/chat_template.h"
#include "core/tokenizer.h"
#include "liyab/mmap_loader.h"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: liyab-tok -m model.gguf -f text.txt [--ids REF] [--bos] [--bench N] [--quiet]\n"
                 "       liyab-tok -m model.gguf (--render REQUEST.json | --format PROBE_JSON)\n"
                 "  --ids REF   compare with reference ids (tools/tokenizer_reference.py); exit 1 on a difference\n"
                 "  --bos       add a BOS by the model's rule (compare with a reference written with --special)\n"
                 "  --bench N   encode the text N times and print the speed\n"
                 "  --quiet     do not print the ids\n"
                 "  --render F  print the chat request in file F rendered with the model's chat template\n"
                 "  --format J  print the template's layout of turns for probe J (JSON; '' for none)\n");
    return 2;
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Reference file: whitespace-separated ids; lines starting with '#' are comments.
bool read_ids(const std::string& path, std::vector<int32_t>& ids) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        long v = 0;
        while (ls >> v) ids.push_back(static_cast<int32_t>(v));
    }
    return true;
}

// A token's text for messages, with control characters escaped.
std::string shown(const liyab::Tokenizer& tok, int32_t id) {
    if (id < 0 || id >= tok.vocab_size()) return "<out of range>";
    const std::string_view text = tok.is_special(id) ? tok.prompt_text(id, false) : std::string_view(tok.piece(id));
    std::string out = "'";
    for (const char c : text) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out + "'";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, text_path, ids_path, render_path, format_probe;
    bool format = false;
    bool bos = false, quiet = false;
    int bench = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "-m" && i + 1 < argc) model_path = next();
        else if (a == "-f" && i + 1 < argc) text_path = next();
        else if (a == "--ids" && i + 1 < argc) ids_path = next();
        else if (a == "--bench" && i + 1 < argc) bench = std::atoi(next());
        else if (a == "--render" && i + 1 < argc) render_path = next();
        else if (a == "--format" && i + 1 < argc) {
            format_probe = next();
            format = true;
        }
        else if (a == "--bos") bos = true;
        else if (a == "--quiet") quiet = true;
        else return usage();
    }
    if (model_path.empty() || (text_path.empty() && render_path.empty() && !format)) return usage();
    if (!render_path.empty() || format) {
        auto file = liyab::MmapLoader::open(model_path);
        if (!file) {
            std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
            return 2;
        }
        auto tok = liyab::Tokenizer::load(**file);
        const auto source = (*file)->get_string("tokenizer.chat_template");
        if (!tok || !source) {
            std::fprintf(stderr, "the model has no %s\n", tok ? "chat template" : "tokenizer");
            return 2;
        }
        auto tmpl = liyab::ChatTemplate::parse(*source);
        if (!tmpl) {
            std::fprintf(stderr, "%s\n", tmpl.status().message().c_str());
            return 1;
        }
        // The special-token variables, as the engine passes them.
        liyab::jinja::Value special = liyab::jinja::Value::object();
        for (const auto& [name, key] : {std::pair{"bos_token", "tokenizer.ggml.bos_token_id"},
                                        std::pair{"eos_token", "tokenizer.ggml.eos_token_id"},
                                        std::pair{"unk_token", "tokenizer.ggml.unknown_token_id"},
                                        std::pair{"pad_token", "tokenizer.ggml.padding_token_id"}}) {
            const auto id = (*file)->get_int(key);
            if (id && *id >= 0 && *id < tok->vocab_size()) {
                special.fields().set(name, liyab::jinja::Value(std::string(tok->token_text(static_cast<int32_t>(*id)))));
            }
        }
        if (format) {
            auto f = tmpl->format(format_probe, special);
            if (!f) {
                std::fprintf(stderr, "%s\n", f.status().message().c_str());
                return 1;
            }
            auto show_field = [](const char* name, const std::string& v) {
                std::printf("%-18s %s\n", name, liyab::jinja::Value(v).repr().c_str());
            };
            std::printf("prefix_stable      %s\n", f->prefix_stable ? "true" : ("false (" + f->reason + ")").c_str());
            show_field("system_prefix", f->system_prefix);
            show_field("system_suffix", f->system_suffix);
            show_field("first_user_prefix", f->first_user_prefix);
            show_field("no_system_head", f->no_system_head);
            show_field("user_prefix", f->user_prefix);
            show_field("user_suffix", f->user_suffix);
            show_field("generation_prompt", f->generation_prompt);
            show_field("assistant_end", f->assistant_end);
            return f->prefix_stable ? 0 : 1;
        }
        std::string request;
        if (!read_file(render_path, request)) {
            std::fprintf(stderr, "cannot read %s\n", render_path.c_str());
            return 2;
        }
        auto out = tmpl->render_chat(request, special);
        if (!out) {
            std::fprintf(stderr, "%s\n", out.status().message().c_str());
            return 1;
        }
        std::fwrite(out->data(), 1, out->size(), stdout);
        return 0;
    }

    std::string text;
    if (!read_file(text_path, text)) {
        std::fprintf(stderr, "cannot read %s\n", text_path.c_str());
        return 2;
    }
    auto file = liyab::MmapLoader::open(model_path);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
        return 2;
    }
    const auto load_start = std::chrono::steady_clock::now();
    auto tok = liyab::Tokenizer::load(**file);
    if (!tok) {
        std::fprintf(stderr, "%s\n", tok.status().to_string().c_str());
        return 2;
    }
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    const bool add_bos = bos && tok->add_bos_default();
    auto encoded = tok->encode(text, add_bos);
    if (!encoded) {
        std::fprintf(stderr, "%s\n", encoded.status().to_string().c_str());
        return 2;
    }
    const std::vector<int32_t>& ids = *encoded;

    if (bench > 0) {
        const auto start = std::chrono::steady_clock::now();
        size_t total = 0;
        for (int r = 0; r < bench; ++r) total += tok->encode(text, add_bos)->size();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("tokenizer loaded in %.1f ms; %zu bytes -> %zu tokens; %.3f ms per encode, %.2f MB/s, %.0f tokens/s\n",
                    load_ms, text.size(), ids.size(), 1e3 * s / bench, text.size() * bench / s / 1e6, total / s);
        return 0;
    }
    if (ids_path.empty()) {
        if (!quiet) {
            for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%c", ids[i], i + 1 == ids.size() ? '\n' : ' ');
        }
        std::printf("%zu tokens\n", ids.size());
        return 0;
    }

    std::vector<int32_t> ref;
    if (!read_ids(ids_path, ref)) {
        std::fprintf(stderr, "cannot read %s\n", ids_path.c_str());
        return 2;
    }
    size_t first = 0;
    while (first < ids.size() && first < ref.size() && ids[first] == ref[first]) ++first;
    if (first == ids.size() && first == ref.size()) {
        std::printf("same ids: %zu tokens\n", ids.size());
        return 0;
    }
    std::printf("tokens: %zu here, %zu in the reference; first difference at token %zu\n", ids.size(), ref.size(), first);
    const size_t from = first >= 3 ? first - 3 : 0;
    for (size_t i = from; i < first + 4 && (i < ids.size() || i < ref.size()); ++i) {
        std::printf("  %5zu  here %7d %-24s  reference %7d %s\n", i, i < ids.size() ? ids[i] : -1,
                    i < ids.size() ? shown(*tok, ids[i]).c_str() : "", i < ref.size() ? ref[i] : -1,
                    i < ref.size() ? shown(*tok, ref[i]).c_str() : "");
    }
    return 1;
}
