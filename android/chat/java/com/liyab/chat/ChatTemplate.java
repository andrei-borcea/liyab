package com.liyab.chat;

import java.util.List;

/**
 * Prompt formats for chat models, picked from the model's vocabulary: a special token that encodes
 * as a single id identifies the family (ChatML for Qwen, Llama 3 headers), otherwise the Zephyr /
 * TinyLlama format is used. Liyab maps special-token texts to their ids when tokenizing.
 *
 * Thinking (ChatML models with <think> tokens, as Qwen3 / Qwen3.5 templates do): ON opens the
 * reasoning block ("<think>\n"), OFF closes it empty ("<think>\n\n</think>\n\n") so the model
 * answers directly; NONE adds nothing. History holds final answers only, without reasoning.
 */
enum ChatTemplate {
    CHATML("ChatML") {
        @Override
        String build(String system, List<String[]> history, String user, Thinking thinking) {
            StringBuilder p = new StringBuilder("<|im_start|>system\n").append(system).append("<|im_end|>\n");
            for (String[] t : history) {
                p.append("<|im_start|>user\n").append(t[0]).append("<|im_end|>\n<|im_start|>assistant\n")
                        .append(t[1]).append("<|im_end|>\n");
            }
            p.append("<|im_start|>user\n").append(user).append("<|im_end|>\n<|im_start|>assistant\n");
            if (thinking == Thinking.ON) p.append("<think>\n");
            else if (thinking == Thinking.OFF) p.append("<think>\n\n</think>\n\n");
            return p.toString();
        }
    },
    LLAMA3("Llama 3") {
        @Override
        String build(String system, List<String[]> history, String user, Thinking thinking) {
            StringBuilder p = new StringBuilder(header("system")).append(system).append("<|eot_id|>");
            for (String[] t : history) {
                p.append(header("user")).append(t[0]).append("<|eot_id|>").append(header("assistant")).append(t[1])
                        .append("<|eot_id|>");
            }
            return p.append(header("user")).append(user).append("<|eot_id|>").append(header("assistant")).toString();
        }

        private String header(String role) {
            return "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
        }
    },
    ZEPHYR("Zephyr") {
        @Override
        String build(String system, List<String[]> history, String user, Thinking thinking) {
            StringBuilder p = new StringBuilder("<|system|>\n").append(system).append("</s>\n");
            for (String[] t : history) {
                p.append("<|user|>\n").append(t[0]).append("</s>\n<|assistant|>\n").append(t[1]).append("</s>\n");
            }
            return p.append("<|user|>\n").append(user).append("</s>\n<|assistant|>\n").toString();
        }
    };

    final String label;

    ChatTemplate(String label) {
        this.label = label;
    }

    enum Thinking { NONE, ON, OFF }

    abstract String build(String system, List<String[]> history, String user, Thinking thinking);

    static ChatTemplate detect(long engine) {
        if (LiyabNative.countTokens(engine, "<|im_start|>") == 1) return CHATML;
        if (LiyabNative.countTokens(engine, "<|start_header_id|>") == 1) return LLAMA3;
        return ZEPHYR;
    }
}
