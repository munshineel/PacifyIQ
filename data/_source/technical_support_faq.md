# Pacify — Technical Support Guide

**Document reference:** FAQ-TEC-001
**Last updated:** 15 January 2026

Work through the section that matches your symptom. If a step asks you to power the
device off, do that properly rather than closing the lid.

---

## S1. Before you start

S1.1 Back up anything you cannot lose. Several steps in this guide, particularly S11, will erase data.

S1.2 Note down any error code you can see, exactly as displayed. Codes are the fastest route to a diagnosis, and S9 lists what each one means.

S1.3 If your device is brand new and has arrived faulty, stop here and report it within 48 hours instead of troubleshooting. That route gets you a free replacement and no fees. See POL-RET-002 S6.

S1.4 It is fine to unbox a new device and switch it on to check it works. Doing so does not stop you returning it, though note that once the retail seal is broken the item counts as opened for the purposes of the return window.

S1.5 Use the charger that came in the box. A large share of the faults reported to us turn out to be a third-party charger.

---

## S2. Will not power on

S2.1 Hold the power button for 15 seconds, release, then press once.

S2.2 Connect the supplied charger and leave it for 30 minutes. A fully depleted battery will not respond immediately.

S2.3 Check the charger light. No light means the charger or the wall socket. Try a different socket.

S2.4 On the ProBook range, hold power plus volume-down for 20 seconds to force a hardware reset.

S2.5 If you see a light or hear a fan but the screen stays dark, that is a display fault, not a power fault. Go to S3.

S2.6 If none of this produces a response, the unit needs service. Raise a warranty claim under POL-WAR-001 S10.

---

## S3. Display problems

S3.1 **Screen is black but the device is running.** Connect an external monitor. If the external picture is fine, the panel or its cable has failed and needs service.

S3.2 **Screen flickers.** Update the display driver first. If flicker persists across a driver reinstall and appears on the login screen as well, it is hardware.

S3.3 **Screen goes dark intermittently on an external monitor.** This is usually a cable or handshake problem rather than a panel fault. Reseat both ends of the cable, try a different cable, and check for a monitor firmware update. Code ERR-DP-0x004 confirms this diagnosis.

S3.4 **Picture is there but the wrong size or refresh rate.** Check the display settings match the panel's native resolution. Code ERR-DP-0x011 means the rate you have selected is not supported over the cable in use.

S3.5 **Dead or stuck pixels.** Count them against a full-screen test at black, white, red, green, and blue. Five or more is covered under warranty; fewer than five is not. See POL-WAR-001 S6.

S3.6 **Backlight is uneven or has gone entirely.** Covered regardless of pixel count. Code DSP-014 indicates backlight failure.

---

## S4. Battery and charging

S4.1 **Not charging.** Try a different socket, then check the cable for damage. Code BAT-042 means the charger is not being recognised, which is usually the cable.

S4.2 **Drains quickly.** Check which applications are consuming power in the system settings. A background sync or an indexing job will flatten a battery in an afternoon.

S4.3 **Check the health figure.** Open Pacify Diagnostics and read the maximum capacity percentage. Below 80% inside the first 12 months is a warranty matter. Above 80%, or beyond 12 months, is normal wear.

S4.4 Code BAT-119 means capacity has fallen below 60%. That is a warranty replacement if the device is under 12 months old — do not attempt further troubleshooting.

S4.5 **Swollen battery or a device that will not sit flat.** Stop using it immediately, unplug it, and contact us. Do not charge it.

---

## S5. Wi-Fi and connectivity

S5.1 Toggle Wi-Fi off and on, then restart the device.

S5.2 Forget the network and rejoin it, entering the password afresh.

S5.3 Restart the router. Two thirds of connectivity reports resolve here.

S5.4 Code WIFI-503 means the wireless driver failed to initialise. Reinstall the driver from the Pacify support site. If it recurs after a clean reinstall, the wireless module needs service.

S5.5 **Connects but no internet.** The problem is the network, not the device. Test with a phone hotspot to confirm.

S5.6 **Bluetooth will not pair.** Remove the existing pairing on both devices and start over. Bluetooth and Wi-Fi share an antenna on the ProBook 14 Lite, so a heavily congested 2.4 GHz environment can affect both.

---

## S6. Audio

S6.1 Check the output device selection first. A previously paired headset will silently claim the output.

S6.2 Code AUD-330 indicates a driver conflict, usually after installing third-party audio software. Uninstall it and reinstall the Pacify audio driver.

S6.3 **One earbud silent on SoundPods.** Place both in the case, close it for 10 seconds, reopen. If it persists, reset by holding both stems for 15 seconds.

S6.4 **Crackling on the internal speakers at high volume** is distortion, not a fault, unless it is present at moderate volume too.

---

## S7. Performance and heat

S7.1 Check for a runaway process in the task manager before anything else.

S7.2 Confirm the vents are clear. A laptop used on bedding will throttle within minutes.

S7.3 Code THRM-88 means the device has been throttling under sustained thermal load. That is protective behaviour, not a fault, unless it occurs at idle.

S7.4 Storage above 90% full will slow the whole system noticeably. Clear space and re-test.

S7.5 Run the Pacify Diagnostics full hardware test. It reports storage health, memory errors, and fan behaviour.

---

## S8. Software and drivers

S8.1 Install pending operating system updates before reporting any software symptom.

S8.2 Drivers are on the Pacify support site under your model. Do not install drivers sourced elsewhere.

S8.3 Note that software faults, operating system errors, driver conflicts, and data loss are excluded from warranty cover under POL-WAR-001 S3.1(f). We will still help you troubleshoot them, but they are not a hardware claim.

S8.4 Installing a different operating system does not void your hardware warranty. Faults caused by it are not covered, but the hardware remains covered.

---

## S9. Error code reference

Codes are shown on the device, in a system notification, or in Pacify Diagnostics. Payment codes beginning PAY- are also listed in POL-PAY-001 S10 with billing-side guidance.

| Code | Meaning | Where it appears | Fix | Warranty? |
|---|---|---|---|---|
| PAY-402 | Payment gateway timeout | Checkout screen | Wait 10 minutes, retry | No |
| PAY-511 | 3-D Secure authentication failed | Bank redirect page | Retry with reachable mobile | No |
| PAY-207 | Insufficient funds | Checkout screen | Different method | No |
| PAY-309 | Card not enabled for online use | Checkout screen | Enable in bank app | No |
| PAY-118 | Declined by issuing bank | Checkout screen | Contact bank | No |
| PAY-604 | Daily limit exceeded | Checkout screen | Retry next day | No |
| ERR-DP-0x004 | DisplayPort handshake failure | Monitor on-screen display | Reseat cable, replace cable, update monitor firmware | No |
| ERR-DP-0x011 | Unsupported refresh rate over current cable | Monitor on-screen display | Lower refresh rate or use DisplayPort | No |
| ERR-HD-0x002 | HDMI signal out of range | Monitor on-screen display | Match output resolution to panel | No |
| BAT-119 | Battery health critical, below 60% | System tray notification | None — service required | Yes, if under 12 months |
| BAT-042 | Charger not recognised | System tray notification | Different cable, then different charger | Charger only |
| BAT-007 | Charging paused, temperature out of range | System tray notification | Let the device cool | No |
| WIFI-503 | Wireless driver initialisation failure | Network settings dialog | Reinstall driver from support site | If it recurs after clean install |
| WIFI-211 | Authentication timeout with access point | Network settings dialog | Rejoin network, restart router | No |
| SYS-0x0000007B | Boot device inaccessible | Boot or stop screen | Check boot order, run storage test | If storage test fails |
| SYS-0x000000EF | Critical process terminated | Stop screen | Repair install of operating system | No |
| THRM-88 | Sustained thermal throttling | Pacify Diagnostics | Clear vents, check ambient temperature | Only if at idle |
| THRM-12 | Fan not responding | Pacify Diagnostics | None — service required | Yes |
| DSP-014 | Panel backlight failure | No code shown, visible symptom | None — service required | Yes |
| DSP-051 | Panel cable fault detected | Pacify Diagnostics | None — service required | Yes |
| AUD-330 | Audio driver conflict | Sound settings dialog | Remove third-party audio software | No |
| KEY-018 | Keyboard controller not responding | Pacify Diagnostics | External keyboard test, then service | Yes |
| STO-440 | Storage health below threshold | Pacify Diagnostics | Back up immediately, then service | Yes |
| MEM-221 | Memory error detected during test | Pacify Diagnostics | None — service required | Yes |
| CAM-090 | Camera module not detected | Device manager | Reinstall driver, then service | If it recurs |

---

## S10. Ports and peripherals

S10.1 Test the peripheral on a second device before assuming the port has failed.

S10.2 USB-C ports on the ProBook range carry data, power, and display. A port that charges but does not carry display is usually a cable limitation — many USB-C cables are power-only.

S10.3 The SD reader on the ProBook 16 supports cards up to 1 TB. Larger cards are not recognised and this is expected.

S10.4 Code KEY-018 on a built-in keyboard requires service. If an external keyboard works, the fault is confirmed as internal.

---

## S11. Factory reset and recovery

S11.1 Back up first. This erases everything.

S11.2 From the operating system, use the built-in reset option and choose whether to keep files.

S11.3 Where the device will not boot, hold power plus the volume-up key during startup to reach the recovery environment.

S11.4 Recovery media for each model is downloadable from the Pacify support site.

S11.5 A reset will not repair a hardware fault. If a symptom survives a clean reset, it is hardware.

---

## S12. When to stop troubleshooting

S12.1 Stop and raise a warranty claim where any of the following is true:

(a) the device shows a code marked "Yes" in the warranty column at S9;
(b) a symptom persists after a factory reset under S11;
(c) an external monitor confirms an internal panel fault under S3.1;
(d) the battery is swollen or the device will not sit flat;
(e) you can smell burning, or the casing is deformed;
(f) a storage or memory test reports errors.

S12.2 Raise the claim under POL-WAR-001 S10. Tell us which steps in this guide you have already tried — it shortens the diagnosis and is not a condition of the claim.

S12.3 If the device is a third-party brand, the claim goes to that manufacturer's service network rather than to Pacify. See POL-WAR-001 S8. We can confirm your purchase date and point you to the nearest centre.

S12.4 If the fault appeared within 48 hours of delivery, do not raise a warranty claim. Report it as a delivery defect under POL-RET-002 S6, which is a better outcome for you.

---

*Pacify Electronics Private Limited. Technical Support, Service Operations.*
