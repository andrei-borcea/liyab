package com.liyab.chat;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.system.Os;
import android.system.OsConstants;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Samples what the running model costs, twice a second, into ring buffers for the overlay charts.
 *
 * Sources an app can actually read on Android (the GPU and battery sysfs nodes are blocked by SELinux,
 * even for adb shell):
 *  - CPU: this process's user + system time from /proc/self/stat, as a share of all cores;
 *  - GPU: the engine's own accounting of time the GPU spent on its work (Engine::counters), i.e. how
 *    busy Liyab keeps the GPU, not other apps;
 *  - power: BatteryManager current x battery voltage. While charging this is the net battery flow, not
 *    the phone's consumption, so it is flagged;
 *  - speed and storage: tokens and streamed bytes from the engine counters; energy per token = W / tok/s.
 */
final class PerfMonitor {
    interface Listener {
        void onSample();  // main thread
    }

    static final int CPU = 0, GPU = 1, POWER = 2, SPEED = 3, FLASH = 4, METRICS = 5;
    static final String[] NAMES = {"CPU", "GPU", "Power", "Speed", "Flash"};
    static final String[] UNITS = {"%", "%", "W", "tok/s", "MB/s"};
    static final float[] FIXED_MAX = {100f, 100f, 0f, 0f, 0f};  // 0: scale to the data
    static final int CAPACITY = 120;  // 60 s at 2 Hz
    static final long PERIOD_MS = 500;

    static final float[][] history = new float[METRICS][CAPACITY];
    static int count;  // valid samples (<= CAPACITY), newest at index count - 1
    static final float[] latest = new float[METRICS];
    static float batteryPercent = -1, batteryTempC = -1, joulesPerToken = -1;
    static boolean charging;
    static int thermalStatus = -1;  // PowerManager.THERMAL_STATUS_*, -1 unknown

    private static final List<Listener> listeners = new CopyOnWriteArrayList<>();
    private static final Handler main = new Handler(Looper.getMainLooper());
    private static Context app;
    private static boolean running;
    private static long lastNanos, lastCpuTicks = -1;
    private static double[] lastCounters;
    private static final long CLK_TCK = Math.max(1, Os.sysconf(OsConstants._SC_CLK_TCK));
    private static final int CORES = Runtime.getRuntime().availableProcessors();

    private static final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            sample();
            for (Listener l : listeners) l.onSample();
            main.postDelayed(this, PERIOD_MS);
        }
    };

    private PerfMonitor() {}

    static void addListener(Listener l) {
        listeners.add(l);
    }

    static void removeListener(Listener l) {
        listeners.remove(l);
    }

    static void start(Context context) {
        if (running) return;
        app = context.getApplicationContext();
        running = true;
        lastNanos = 0;
        main.post(tick);
    }

    static void stop() {
        running = false;
        main.removeCallbacks(tick);
    }

    private static void sample() {
        final long now = System.nanoTime();
        final double dt = lastNanos == 0 ? 0 : (now - lastNanos) / 1e9;
        lastNanos = now;
        float[] v = new float[METRICS];

        long ticks = cpuTicks();
        if (dt > 0 && ticks >= 0 && lastCpuTicks >= 0) {
            v[CPU] = (float) Math.min(100.0, 100.0 * (ticks - lastCpuTicks) / CLK_TCK / dt / CORES);
        }
        lastCpuTicks = ticks;

        double[] c = null;
        synchronized (EngineHolder.LIFECYCLE) {
            if (EngineHolder.handle != 0) c = LiyabNative.counters(EngineHolder.handle);
        }
        if (dt > 0 && c != null && lastCounters != null && c[2] >= lastCounters[2]) {
            v[GPU] = (float) Math.min(100.0, (c[0] - lastCounters[0]) / (dt * 10.0));  // ms per s -> %
            v[FLASH] = (float) ((c[1] - lastCounters[1]) / dt / 1e6);
            v[SPEED] = (float) ((c[2] - lastCounters[2]) / dt);
        }
        lastCounters = c;

        readBattery(v);
        joulesPerToken = v[SPEED] > 0.2f && v[POWER] > 0 ? v[POWER] / v[SPEED] : -1;

        if (dt <= 0) return;  // first call: only establishes the baselines
        for (int m = 0; m < METRICS; ++m) {
            if (count == CAPACITY) System.arraycopy(history[m], 1, history[m], 0, CAPACITY - 1);
            history[m][Math.min(count, CAPACITY - 1)] = v[m];
            latest[m] = v[m];
        }
        if (count < CAPACITY) ++count;
    }

    /** utime + stime of this process in clock ticks, or -1. */
    private static long cpuTicks() {
        try (FileInputStream in = new FileInputStream("/proc/self/stat")) {
            byte[] buf = new byte[1024];
            int n = in.read(buf);
            if (n <= 0) return -1;
            String s = new String(buf, 0, n, StandardCharsets.US_ASCII);
            // Fields after the parenthesized command: state is field 3, utime 14, stime 15.
            String[] f = s.substring(s.lastIndexOf(')') + 2).split(" ");
            return Long.parseLong(f[11]) + Long.parseLong(f[12]);
        } catch (IOException | RuntimeException e) {
            return -1;
        }
    }

    private static void readBattery(float[] v) {
        Intent battery = app.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        BatteryManager bm = (BatteryManager) app.getSystemService(Context.BATTERY_SERVICE);
        if (battery == null || bm == null) return;
        int level = battery.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
        int scale = battery.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
        batteryPercent = level >= 0 ? 100f * level / Math.max(1, scale) : -1;
        batteryTempC = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -10) / 10f;
        charging = battery.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0;
        int millivolts = battery.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0);
        long current = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
        if (current != Integer.MIN_VALUE && millivolts > 0) {
            // The API says microamperes, but some devices report milliamperes.
            double amps = Math.abs(current) >= 20000 ? Math.abs(current) / 1e6 : Math.abs(current) / 1e3;
            v[POWER] = (float) (amps * millivolts / 1000.0);
        }
        PowerManager pm = (PowerManager) app.getSystemService(Context.POWER_SERVICE);
        if (pm != null && android.os.Build.VERSION.SDK_INT >= 29) thermalStatus = pm.getCurrentThermalStatus();
    }
}
