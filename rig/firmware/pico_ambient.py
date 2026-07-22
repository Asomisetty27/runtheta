"""
MicroPython firmware for the Raspberry Pi Pico (RP2040) -- the measured-ambient half
of the E-MR rig log. Flash with Thonny or `mpremote` (save as main.py on the Pico).

Reads DS18B20 air sensors (inlet / exhaust / room), drives the 4-pin PWM fan (the
degradation knob for the airflow arm), reads the fan tach, and streams one CSV line per
second over USB serial:

    elapsed_ms,inlet_c,exhaust_c,case_c,fan_duty,fan_rpm

The host (rig/serial_logger.py) timestamps each line with host epoch and joins it onto
the NVML capture by time. Optional: send "D<pct>\\n" over serial to set the fan duty
(runs the airflow/fan arm from the host).

Wiring: DS18B20 data -> GP2 with a 4.7k pull-up to 3V3 (all sensors share the bus; the
scan() order maps to inlet/exhaust/case by which you connect first). Fan PWM -> GP0,
fan tach -> GP1 (pull-up). Fan power is 12 V from the PSU with a COMMON GROUND to the
Pico -- never power the fan from the Pico.
"""
import sys
import time

import machine
import onewire
import ds18x20

DS_PIN = 2
FAN_PWM_PIN = 0
FAN_TACH_PIN = 1
PWM_FREQ = 25000     # 25 kHz: the standard for 4-pin PWM fans
SAMPLE_MS = 1000
TACH_PULSES_PER_REV = 2

ow = onewire.OneWire(machine.Pin(DS_PIN))
ds = ds18x20.DS18X20(ow)
roms = ds.scan()     # sensor addresses; wiring order = inlet, exhaust, room/case

fan = machine.PWM(machine.Pin(FAN_PWM_PIN))
fan.freq(PWM_FREQ)
_duty = 100


def set_fan(pct):
    global _duty
    _duty = max(0, min(100, int(pct)))
    fan.duty_u16(int(_duty * 65535 / 100))


set_fan(100)

_tach = 0


def _tick(_pin):
    global _tach
    _tach += 1


machine.Pin(FAN_TACH_PIN, machine.Pin.IN, machine.Pin.PULL_UP).irq(
    trigger=machine.Pin.IRQ_FALLING, handler=_tick)

poll = machine.Pin(FAN_TACH_PIN)  # keep a ref so the IRQ isn't GC'd


def _maybe_read_command():
    """Non-blocking: honor a 'D<pct>' fan-duty command from the host if one is waiting."""
    try:
        if hasattr(sys.stdin, "any") and sys.stdin.any():
            line = sys.stdin.readline().strip()
            if line and line[0] == "D":
                set_fan(int(line[1:]))
    except Exception:
        pass


def main():
    global _tach
    t0 = time.ticks_ms()
    last = t0
    while True:
        ds.convert_temp()
        time.sleep_ms(750)               # DS18B20 conversion time at 12-bit
        temps = []
        for r in roms:
            try:
                temps.append("%.3f" % ds.read_temp(r))
            except Exception:
                temps.append("")
        while len(temps) < 3:
            temps.append("")

        now = time.ticks_ms()
        dt_s = time.ticks_diff(now, last) / 1000
        last = now
        rpm = int((_tach / TACH_PULSES_PER_REV) / dt_s * 60) if dt_s > 0 else 0
        _tach = 0
        elapsed = time.ticks_diff(now, t0)

        print("%d,%s,%s,%s,%d,%d" % (elapsed, temps[0], temps[1], temps[2], _duty, rpm))
        _maybe_read_command()
        time.sleep_ms(max(0, SAMPLE_MS - 750))


if __name__ == "__main__":
    main()
