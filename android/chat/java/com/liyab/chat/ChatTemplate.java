package com.liyab.chat;

import java.util.List;

/**
 * Prompt formats for chat models, picked from the model's vocabulary: a special token that encodes
 * as a single id identifies the family (ChatML for Qwen, Llama 3 headers), otherwise the Zephyr /
 * TinyLlama format is used. Liyab maps special-token texts to their ids when tokenizing.
 *
 * Thinking (ChatML models with <think> tokens, as Qwen3 / Qwen3.5 templates do): ON opens the
 * reasoning block ("<think>\n"), OFF closes it empty ("<think>\n\n</think>\n\n") so the model
 * answers directly; NONE adds nothing. Past turns are replayed exactly as generated (reasoning
 * included) so the engine can keep its context between turns instead of reprocessing the chat.
 */
enum ChatTemplate {
    CHATML("ChatML") {
        @Override
        String system(String system) {
            return "<|im_start|>system\n" + system + "<|im_end|>\n";
        }

        @Override
        String turn(String user, String assistant) {
            return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + assistant + "<|im_end|>\n";
        }

        @Override
        String open(String user, Thinking thinking) {
            return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n" + assistantPrefix(thinking);
        }

        @Override
        String assistantPrefix(Thinking thinking) {
            return thinking == Thinking.ON ? "<think>\n" : thinking == Thinking.OFF ? "<think>\n\n</think>\n\n" : "";
        }
    },
    LLAMA3("Llama 3") {
        @Override
        String system(String system) {
            return header("system") + system + "<|eot_id|>";
        }

        @Override
        String turn(String user, String assistant) {
            return header("user") + user + "<|eot_id|>" + header("assistant") + assistant + "<|eot_id|>";
        }

        @Override
        String open(String user, Thinking thinking) {
            return header("user") + user + "<|eot_id|>" + header("assistant");
        }

        private String header(String role) {
            return "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
        }
    },
    ZEPHYR("Zephyr") {
        @Override
        String system(String system) {
            return "<|system|>\n" + system + "</s>\n";
        }

        @Override
        String turn(String user, String assistant) {
            return "<|user|>\n" + user + "</s>\n<|assistant|>\n" + assistant + "</s>\n";
        }

        @Override
        String open(String user, Thinking thinking) {
            return "<|user|>\n" + user + "</s>\n<|assistant|>\n";
        }
    };

    final String label;

    ChatTemplate(String label) {
        this.label = label;
    }

    enum Thinking { NONE, ON, OFF }

    /** The system block that starts every prompt (prefilled right after loading). */
    abstract String system(String system);

    /** One finished turn: the user message and the assistant text exactly as the model produced it. */
    abstract String turn(String user, String assistant);

    /** The new user message and the opening of the assistant reply the model continues. */
    abstract String open(String user, Thinking thinking);

    /** What open() puts at the start of the assistant reply (ChatML thinking control), else "". */
    String assistantPrefix(Thinking thinking) {
        return "";
    }

    /**
     * The full prompt. History entries are {user, answer, exact}: `exact` (assistantPrefix + the raw
     * generated text) rebuilds each past turn token for token as the model saw it, so every new prompt
     * continues the engine's context and only the new message is processed. Older entries without
     * `exact` fall back to the answer.
     */
    String build(String system, List<String[]> history, String user, Thinking thinking) {
        StringBuilder p = new StringBuilder(system(system));
        for (String[] t : history) p.append(turn(t[0], t.length > 2 ? t[2] : t[1]));
        return p.append(open(user, thinking)).toString();
    }

    static ChatTemplate detect(long engine) {
        if (LiyabNative.countTokens(engine, "<|im_start|>") == 1) return CHATML;
        if (LiyabNative.countTokens(engine, "<|start_header_id|>") == 1) return LLAMA3;
        return ZEPHYR;
    }
}
