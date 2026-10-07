package com.liyab.chat;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Minimal Hugging Face Hub client: GGUF model search, file listing and resumable downloads.
 * Uses only the public, unauthenticated API (https://huggingface.co/api/...). All methods block
 * and must run off the main thread.
 */
final class HuggingFace {
    private static final String API = "https://huggingface.co";
    private static final String USER_AGENT = "LiyabChat/0.1 (Android)";
    /** Architectures the engine runs (keep in sync with src/core/transformer.cpp). */
    static final java.util.Set<String> SUPPORTED_ARCHS =
            new java.util.TreeSet<>(java.util.Arrays.asList("llama", "mistral", "qwen2", "qwen3", "qwen35"));
    // One part of a split GGUF model (gguf-split): <prefix>-00002-of-00005.gguf
    private static final Pattern SPLIT = Pattern.compile("-(\\d{5})-of-(\\d{5})\\.gguf$", Pattern.CASE_INSENSITIVE);
    private static final Pattern QUANT = Pattern.compile(
            "(IQ\\d_[A-Z]+|Q\\d_K(?:_[SML])?|Q\\d_\\d|TQ\\d_\\d|BF16|F16|F32|MXFP4|NVFP4)", Pattern.CASE_INSENSITIVE);

    /** Whether Liyab can load a file, judged from its quantization name. */
    enum Compat {
        OK("✓ supported"),
        UNSUPPORTED("✗ format not supported");

        final String label;

        Compat(String label) {
            this.label = label;
        }
    }

    static final class Repo {
        final String id;
        final long downloads;
        final long likes;

        Repo(String id, long downloads, long likes) {
            this.id = id;
            this.downloads = downloads;
            this.likes = likes;
        }
    }

    static final class GgufFile {
        final String repo;
        final String path;
        final long size;
        final String sha256;  // from the LFS pointer; null when unknown
        final String quant;
        final Compat compat;
        final boolean split;  // one part of a multi-file GGUF (-00001-of-00003.gguf)
        final boolean auxiliary;  // not a language model: mmproj (vision), imatrix, MTP heads
        // A split model as a whole: its parts in order (path = part 1, size = total); empty otherwise.
        final List<GgufFile> parts;

        GgufFile(String repo, String path, long size) {
            this(repo, path, size, null);
        }

        GgufFile(String repo, String path, long size, String sha256) {
            this(repo, path, size, sha256, java.util.Collections.emptyList());
        }

        private GgufFile(String repo, String path, long size, String sha256, List<GgufFile> parts) {
            this.repo = repo;
            this.path = path;
            this.size = size;
            this.sha256 = sha256;
            this.parts = parts;
            this.split = parts.isEmpty() && SPLIT.matcher(path).find();
            String lower = path.toLowerCase(Locale.US);
            this.auxiliary = lower.contains("mmproj") || lower.contains("imatrix") || lower.startsWith("mtp/")
                    || lower.contains("/mtp-");
            Matcher m = QUANT.matcher(path);
            String q = null;
            while (m.find()) q = m.group(1).toUpperCase(Locale.US);  // last match: the suffix
            this.quant = q != null ? q : "?";
            this.compat = split || auxiliary ? Compat.UNSUPPORTED : compatOf(this.quant);
        }

        /** The whole model made of `parts` (all parts of one split set, in order). */
        static GgufFile splitModel(List<GgufFile> parts) {
            long total = 0;
            for (GgufFile p : parts) total += p.size;
            GgufFile first = parts.get(0);
            return new GgufFile(first.repo, first.path, total, null, parts);
        }

        String fileName() {
            int slash = path.lastIndexOf('/');
            return slash >= 0 ? path.substring(slash + 1) : path;
        }

        /** Whether every byte of this model (all parts) is in `dir`. */
        boolean presentIn(File dir) {
            if (parts.isEmpty()) return new File(dir, fileName()).length() == size;
            for (GgufFile p : parts) if (!p.presentIn(dir)) return false;
            return true;
        }
    }

    /** Path of part 1 of the split set `path` belongs to, or `path` itself for a single file. */
    static String firstPart(String path) {
        Matcher m = SPLIT.matcher(path);
        if (!m.find()) return path;
        return path.substring(0, m.start()) + "-00001-of-" + m.group(2) + ".gguf";
    }

    /** The files of the model whose first (or only) file is `first`, in order; just `first` if not split. */
    static List<File> localParts(File first) {
        Matcher m = SPLIT.matcher(first.getName());
        if (!m.find() || !m.group(1).equals("00001")) return java.util.Collections.singletonList(first);
        int count = Integer.parseInt(m.group(2));
        String prefix = first.getName().substring(0, m.start());
        List<File> out = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            out.add(new File(first.getParentFile(), String.format(Locale.US, "%s-%05d-of-%05d.gguf", prefix, i, count)));
        }
        return out;
    }

    /** Whether `name` is part 2+ of a split model (listed through its part 1). */
    static boolean isLaterPart(String name) {
        Matcher m = SPLIT.matcher(name);
        return m.find() && !m.group(1).equals("00001");
    }

    /** Live download state, polled by the UI. All fields are updated by download threads. */
    static final class DownloadState {
        final java.util.concurrent.atomic.AtomicLong done = new java.util.concurrent.atomic.AtomicLong();
        volatile long total;
        final java.util.concurrent.atomic.AtomicInteger activeConnections = new java.util.concurrent.atomic.AtomicInteger();
        final java.util.concurrent.atomic.AtomicInteger retries = new java.util.concurrent.atomic.AtomicInteger();
        volatile String lastError = "";  // most recent transient error, shown while retrying
        volatile boolean verifying;      // SHA-256 check after the last byte arrived
        final java.util.concurrent.atomic.AtomicLong verified = new java.util.concurrent.atomic.AtomicLong();
        volatile boolean cancelled;      // set by the UI: stop and keep the partial file
        volatile long base;              // bytes of earlier parts already counted in `done` (split models)
    }

    private HuggingFace() {}

    /**
     * Liyab decodes every GGML tensor format llama.cpp ships (F32/F16/BF16, Q4_0..Q8_0, Q2_K..Q6_K and
     * their _S/_M/_L mixes, all IQ formats, TQ1_0/TQ2_0, MXFP4, NVFP4) except Q1_0. Unrecognised names
     * (no quantization in the file name) are listed as supported; the loader reports exact errors.
     */
    static Compat compatOf(String quant) {
        return quant.equals("Q1_0") ? Compat.UNSUPPORTED : Compat.OK;
    }

    /** GGUF repositories matching `query` (empty: most downloaded), best first. */
    static List<Repo> search(String query) throws IOException {
        String url = API + "/api/models?filter=gguf&sort=downloads&direction=-1&limit=40"
                + (query.isEmpty() ? "" : "&search=" + URLEncoder.encode(query, "UTF-8"));
        try {
            JSONArray array = new JSONArray(getString(url));
            List<Repo> repos = new ArrayList<>();
            for (int i = 0; i < array.length(); i++) {
                JSONObject o = array.getJSONObject(i);
                repos.add(new Repo(o.getString("id"), o.optLong("downloads"), o.optLong("likes")));
            }
            return repos;
        } catch (org.json.JSONException e) {
            throw new IOException("unexpected Hugging Face response: " + e.getMessage());
        }
    }

    /**
     * Reads `general.architecture` from the start of `file` with an HTTP range request (it is
     * among the first metadata keys), so unsupported models are flagged before downloading GBs.
     * Returns null when it cannot be determined.
     */
    static String architecture(GgufFile file) {
        try {
            HttpURLConnection c = open(API + "/" + file.repo + "/resolve/main/" + encodePath(file.path));
            c.setRequestProperty("Range", "bytes=0-65535");
            try (InputStream in = c.getInputStream()) {
                byte[] head = readAll(in);
                java.nio.ByteBuffer b = java.nio.ByteBuffer.wrap(head).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                if (b.getInt() != 0x46554747) return null;  // "GGUF"
                b.getInt();   // version
                b.getLong();  // tensor count
                long kvs = b.getLong();
                for (long i = 0; i < kvs && b.remaining() > 12; i++) {
                    String key = readString(b);
                    int type = b.getInt();
                    if (key.equals("general.architecture") && type == 8) return readString(b);
                    if (!skipValue(b, type)) return null;
                }
            } finally {
                c.disconnect();
            }
        } catch (IOException | RuntimeException e) {
            return null;
        }
        return null;
    }

    private static String readString(java.nio.ByteBuffer b) {
        int n = (int) b.getLong();
        byte[] s = new byte[n];
        b.get(s);
        return new String(s, StandardCharsets.UTF_8);
    }

    // Skips one metadata value of GGUF type `type`; false if it does not fit in the buffer.
    private static boolean skipValue(java.nio.ByteBuffer b, int type) {
        final int[] sizes = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type == 8) {
            long n = b.getLong();
            if (n > b.remaining()) return false;
            b.position(b.position() + (int) n);
            return true;
        }
        if (type == 9) {
            int elem = b.getInt();
            long count = b.getLong();
            for (long i = 0; i < count; i++) {
                if (b.remaining() < 8 || !skipValue(b, elem)) return false;
            }
            return true;
        }
        if (type < 0 || type >= sizes.length || sizes[type] == 0 || b.remaining() < sizes[type]) return false;
        b.position(b.position() + sizes[type]);
        return true;
    }

    /** The .gguf files of `repo` with their sizes, smallest first. */
    static List<GgufFile> files(String repo) throws IOException {
        List<GgufFile> files = new ArrayList<>();
        try {
            JSONArray array = new JSONArray(getString(API + "/api/models/" + repo + "/tree/main?recursive=true"));
            for (int i = 0; i < array.length(); i++) {
                JSONObject o = array.getJSONObject(i);
                String path = o.getString("path");
                if (!"file".equals(o.optString("type")) || !path.toLowerCase(Locale.US).endsWith(".gguf")) continue;
                JSONObject lfs = o.optJSONObject("lfs");
                long size = lfs != null ? lfs.optLong("size", o.optLong("size")) : o.optLong("size");
                String sha = lfs != null ? lfs.optString("oid", null) : null;
                files.add(new GgufFile(repo, path, size, sha));
            }
        } catch (org.json.JSONException e) {
            throw new IOException("unexpected Hugging Face response: " + e.getMessage());
        }
        // Split sets become one entry once every part is listed; their parts are not shown alone.
        java.util.Map<String, List<GgufFile>> sets = new java.util.TreeMap<>();
        for (GgufFile f : files) if (f.split) sets.computeIfAbsent(firstPart(f.path), k -> new ArrayList<>()).add(f);
        for (List<GgufFile> set : sets.values()) {
            set.sort((a, b) -> a.path.compareTo(b.path));
            Matcher m = SPLIT.matcher(set.get(0).path);
            if (m.find() && set.size() == Integer.parseInt(m.group(2))) files.add(GgufFile.splitModel(set));
        }
        files.sort((a, b) -> Long.compare(a.size, b.size));
        return files;
    }

    static final int SEGMENTS = 4;      // parallel HTTP range connections
    static final int MAX_ATTEMPTS = 8;  // per segment, exponential backoff 1 s .. 64 s

    /**
     * Downloads `file` into `dir` with SEGMENTS parallel range requests. Progress of every segment is
     * persisted in `<name>.part.state`, so a cancelled, crashed or interrupted download resumes where
     * it stopped; transient network errors are retried per segment with exponential backoff without
     * losing received bytes. Returns the finished file; throws on cancellation or when a segment keeps
     * failing (the partial file and its state are kept for the next attempt).
     */
    static File downloadModel(GgufFile model, File dir, DownloadState state) throws IOException {
        if (model.parts.isEmpty()) return download(model, dir, state);
        state.total = model.size;
        long base = 0;
        for (GgufFile part : model.parts) {
            state.base = base;
            download(part, dir, state);
            base += part.size;
        }
        return new File(dir, model.fileName());  // part 1: the engine opens the others next to it
    }

    /** Downloads one file (a model, or one part of a split model; see downloadModel). */
    static File download(GgufFile file, File dir, DownloadState state) throws IOException {
        File target = new File(dir, file.fileName());
        if (target.exists() && target.length() == file.size) {
            state.done.set(state.base + file.size);
            return target;
        }
        File part = new File(dir, file.fileName() + ".part");
        File stateFile = new File(dir, file.fileName() + ".part.state");
        final long size = file.size;
        if (size <= 0) throw new IOException("unknown file size");

        // Segment i covers [start[i], end[i]); progress[i] bytes of it are on disk.
        final long[] start = new long[SEGMENTS];
        final long[] end = new long[SEGMENTS];
        final long[] progress = new long[SEGMENTS];
        for (int i = 0; i < SEGMENTS; i++) {
            start[i] = size * i / SEGMENTS;
            end[i] = size * (i + 1) / SEGMENTS;
        }
        if (part.exists() && part.length() == size && stateFile.exists()) {
            loadState(stateFile, size, progress);
        } else {
            stateFile.delete();
        }
        long remaining = 0;
        for (int i = 0; i < SEGMENTS; i++) remaining += end[i] - start[i] - progress[i];
        if (dir.getUsableSpace() < (part.exists() ? remaining : size)) {
            throw new IOException(String.format(Locale.US, "not enough free space: need %.2f GB, have %.2f GB",
                    remaining / 1e9, dir.getUsableSpace() / 1e9));
        }
        try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(part, "rw")) {
            raf.setLength(size);  // pre-size so segments can be written independently
        }
        writeSource(new File(dir, file.fileName() + ".part.src"), file);
        long already = 0;
        for (long p : progress) already += p;
        if (state.total == 0) state.total = size;
        state.done.set(state.base + already);

        final String url = API + "/" + file.repo + "/resolve/main/" + encodePath(file.path);
        final IOException[] failure = new IOException[1];
        Thread[] workers = new Thread[SEGMENTS];
        Thread saver = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                synchronized (progress) {
                    saveState(stateFile, size, progress);
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        saver.start();
        for (int i = 0; i < SEGMENTS; i++) {
            final int seg = i;
            workers[i] = new Thread(() -> {
                try {
                    downloadSegment(url, part, start[seg], end[seg], progress, seg, state);
                } catch (IOException e) {
                    synchronized (failure) {
                        if (failure[0] == null) failure[0] = e;
                    }
                    state.cancelled = true;  // stop the other segments too
                }
            }, "hf-download-" + i);
            workers[i].start();
        }
        for (Thread t : workers) {
            try {
                t.join();
            } catch (InterruptedException e) {
                state.cancelled = true;
                Thread.currentThread().interrupt();
            }
        }
        saver.interrupt();
        synchronized (progress) {
            saveState(stateFile, size, progress);
        }
        long done = 0;
        for (long p : progress) done += p;
        if (done != size) {
            if (failure[0] != null) throw failure[0];
            throw new IOException("download paused");
        }
        if (file.sha256 != null && file.sha256.length() == 64) {
            state.verifying = true;
            String actual = sha256(part, state);
            if (state.cancelled) throw new IOException("download paused");
            if (!actual.equalsIgnoreCase(file.sha256)) {
                discard(dir, file);  // corrupt: start over next time
                throw new IOException("checksum mismatch (expected " + file.sha256.substring(0, 12) + "…, got "
                        + actual.substring(0, 12) + "…); the file was deleted, download it again");
            }
        }
        stateFile.delete();
        new File(dir, file.fileName() + ".part.src").delete();
        if (!part.renameTo(target)) throw new IOException("cannot rename " + part + " to " + target);
        return target;
    }

    private static String sha256(File f, DownloadState state) throws IOException {
        java.security.MessageDigest md;
        try {
            md = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
        byte[] buffer = new byte[1 << 20];
        try (InputStream in = new java.io.FileInputStream(f)) {
            int n;
            while (!state.cancelled && (n = in.read(buffer)) > 0) {
                md.update(buffer, 0, n);
                state.verified.addAndGet(n);
            }
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : md.digest()) hex.append(String.format(Locale.US, "%02x", b));
        return hex.toString();
    }

    /** An unfinished download found in `dir`, with its progress. */
    static final class Partial {
        final GgufFile file;
        final long done;

        Partial(GgufFile file, long done) {
            this.file = file;
            this.done = done;
        }
    }

    static List<Partial> partials(File dir) {
        List<Partial> out = new ArrayList<>();
        File[] sources = dir.listFiles((d, name) -> name.endsWith(".part.src"));
        if (sources == null) return out;
        for (File src : sources) {
            try (InputStream in = new java.io.FileInputStream(src)) {
                String[] lines = new String(readAll(in), StandardCharsets.UTF_8).split("\n");
                GgufFile f = new GgufFile(lines[0], lines[1], Long.parseLong(lines[2].trim()),
                        lines.length > 3 && !lines[3].trim().isEmpty() ? lines[3].trim() : null);
                long[] progress = new long[SEGMENTS];
                loadState(new File(dir, f.fileName() + ".part.state"), f.size, progress);
                long done = 0;
                for (long p : progress) done += p;
                out.add(new Partial(f, done));
            } catch (IOException | RuntimeException ignored) {
                // unreadable source: ignore
            }
        }
        return out;
    }

    /** Deletes a partial download and its bookkeeping. */
    static void discard(File dir, GgufFile file) {
        for (String suffix : new String[] {".part", ".part.state", ".part.src", ".part.state.tmp"}) {
            new File(dir, file.fileName() + suffix).delete();
        }
    }

    private static void writeSource(File src, GgufFile file) {
        try (OutputStream out = new FileOutputStream(src)) {
            out.write((file.repo + "\n" + file.path + "\n" + file.size + "\n" + (file.sha256 != null ? file.sha256 : "")
                    + "\n").getBytes(StandardCharsets.UTF_8));
        } catch (IOException ignored) {
            // resume from the picker will not be offered, but the download itself still works
        }
    }

    private static void downloadSegment(String url, File part, long start, long end, long[] progress, int seg,
                                        DownloadState state) throws IOException {
        int attempt = 0;
        byte[] buffer = new byte[256 * 1024];
        while (true) {
            long offset;
            synchronized (progress) {
                offset = start + progress[seg];
            }
            if (offset >= end || state.cancelled) return;
            HttpURLConnection c = null;
            try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(part, "rw")) {
                c = open(url);
                c.setRequestProperty("Range", "bytes=" + offset + "-" + (end - 1));
                int code = c.getResponseCode();
                if (code != 206) {
                    if (code == 404 || code == 401 || code == 403) throw new FatalHttp("HTTP " + code);
                    throw new IOException("HTTP " + code + " (range not honoured)");
                }
                raf.seek(offset);
                state.activeConnections.incrementAndGet();
                try (InputStream in = c.getInputStream()) {
                    int n;
                    while (!state.cancelled && offset < end && (n = in.read(buffer, 0, (int) Math.min(buffer.length, end - offset))) > 0) {
                        raf.write(buffer, 0, n);
                        offset += n;
                        synchronized (progress) {
                            progress[seg] += n;
                        }
                        state.done.addAndGet(n);
                        attempt = 0;  // progress made: reset the backoff
                    }
                } finally {
                    state.activeConnections.decrementAndGet();
                }
            } catch (FatalHttp e) {
                throw new IOException(e.getMessage() + " for " + url);
            } catch (IOException e) {
                if (state.cancelled) return;
                if (++attempt >= MAX_ATTEMPTS) throw new IOException("segment " + seg + " failed " + attempt + " times: " + e.getMessage());
                state.retries.incrementAndGet();
                state.lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
                long backoff = Math.min(64_000L, 1000L << (attempt - 1));
                long until = System.currentTimeMillis() + backoff;
                while (!state.cancelled && System.currentTimeMillis() < until) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException ie) {
                        return;
                    }
                }
            } finally {
                if (c != null) c.disconnect();
            }
        }
    }

    /** HTTP errors that retrying cannot fix. */
    private static final class FatalHttp extends IOException {
        FatalHttp(String message) {
            super(message);
        }
    }

    private static void saveState(File stateFile, long size, long[] progress) {
        StringBuilder sb = new StringBuilder().append(size);
        for (long p : progress) sb.append(' ').append(p);
        File tmp = new File(stateFile.getPath() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(tmp)) {
            out.write(sb.toString().getBytes(StandardCharsets.US_ASCII));
            out.getFD().sync();  // the state must never claim bytes that are not on disk yet
        } catch (IOException ignored) {
            return;
        }
        tmp.renameTo(stateFile);  // atomic replace: never a half-written state
    }

    private static void loadState(File stateFile, long size, long[] progress) {
        try (InputStream in = new java.io.FileInputStream(stateFile)) {
            String[] parts = new String(readAll(in), StandardCharsets.US_ASCII).trim().split(" ");
            if (parts.length != SEGMENTS + 1 || Long.parseLong(parts[0]) != size) return;
            for (int i = 0; i < SEGMENTS; i++) {
                long segLen = size * (i + 1) / SEGMENTS - size * i / SEGMENTS;
                progress[i] = Math.max(0, Math.min(segLen, Long.parseLong(parts[i + 1])));
            }
        } catch (IOException | NumberFormatException ignored) {
            java.util.Arrays.fill(progress, 0);
        }
    }

    private static String encodePath(String path) throws IOException {
        StringBuilder encoded = new StringBuilder();
        for (String segment : path.split("/")) {
            if (encoded.length() > 0) encoded.append('/');
            encoded.append(URLEncoder.encode(segment, "UTF-8").replace("+", "%20"));
        }
        return encoded.toString();
    }

    private static HttpURLConnection open(String url) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setRequestProperty("User-Agent", USER_AGENT);
        c.setConnectTimeout(15_000);
        c.setReadTimeout(60_000);
        c.setInstanceFollowRedirects(true);  // /resolve redirects to the CDN (https -> https)
        return c;
    }

    private static String getString(String url) throws IOException {
        HttpURLConnection c = open(url);
        try {
            if (c.getResponseCode() != 200) throw new IOException("HTTP " + c.getResponseCode() + " for " + url);
            try (InputStream in = c.getInputStream()) {
                byte[] data = readAll(in);
                return new String(data, StandardCharsets.UTF_8);
            }
        } finally {
            c.disconnect();
        }
    }

    private static byte[] readAll(InputStream in) throws IOException {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[64 * 1024];
        int n;
        while ((n = in.read(buffer)) > 0) out.write(buffer, 0, n);
        return out.toByteArray();
    }
}
