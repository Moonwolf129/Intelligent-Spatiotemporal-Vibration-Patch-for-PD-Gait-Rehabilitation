import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class VibrationCommand:
    t_start: float
    t_end: float
    channels: List[int]       # e.g. [1, 2, 3, 4]
    amplitude: float          # 0..1 (to be mapped to PWM)
    frequency: float          # Hz (20 or 40)
    mode: str = "SpaVib"      # "SpaVib" or "Const"


@dataclass
class VibroControllerParams:
    freq_hz: float = 40.0
    amplitude: float = 1.0
    trigger_angle_deg: float = 15.0  # angle where leg-lift initiation starts
    trigger_slope_deg_per_s: float = 60.0
    pulse_duration_s: float = 0.2
    refractory_s: float = 0.6        # minimum time between cues


class VibrationController:
    """
    Generate spatiotemporal vibration commands from fused thigh angle
    and target template.

    Concept:
    - When the fused thigh angle enters a rapid upswing region
      (angle around trigger_angle and slope above threshold),
      trigger a short vibration burst just before swing initiation.
    - The returned commands can be mapped to PWM and BLE packets
      on the MCU and smartphone.
    """

    def __init__(self, params: VibroControllerParams = None):
        self.params = params or VibroControllerParams()

    def generate_commands(self,
                          theta: np.ndarray,
                          t: np.ndarray,
                          mode: str = "SpaVib") -> List[VibrationCommand]:
        """
        Generate vibration commands for an entire walking sequence.

        Parameters
        ----------
        theta : (N,) array
            Thigh angle trajectory (deg).
        t : (N,) array
        mode : str
            "SpaVib" or "Const".

        Returns
        -------
        commands : list of VibrationCommand
        """
        cmds: List[VibrationCommand] = []
        N = len(theta)
        dt = np.diff(t, prepend=t[0])
        dtheta = np.gradient(theta, t)
        last_trigger_time = -np.inf

        for i in range(1, N):
            time = t[i]
            slope = dtheta[i]
            angle = theta[i]

            if time - last_trigger_time < self.params.refractory_s:
                continue

            if mode == "SpaVib":
                # Pre-swing: angle near trigger_angle and rising fast
                if angle >= self.params.trigger_angle_deg and \
                        slope >= self.params.trigger_slope_deg_per_s:
                    cmd = VibrationCommand(
                        t_start=time,
                        t_end=time + self.params.pulse_duration_s,
                        channels=[1, 2, 3, 4],
                        amplitude=self.params.amplitude,
                        frequency=self.params.freq_hz,
                        mode="SpaVib",
                    )
                    cmds.append(cmd)
                    last_trigger_time = time
            elif mode == "Const":
                # Constant vibration can be generated outside this class
                # (e.g. fixed duty cycle), but we keep an example:
                if not cmds:
                    cmd = VibrationCommand(
                        t_start=t[0],
                        t_end=t[-1],
                        channels=[1, 2, 3, 4],
                        amplitude=self.params.amplitude,
                        frequency=self.params.freq_hz,
                        mode="Const",
                    )
                    cmds.append(cmd)
                    break

        return cmds

    @staticmethod
    def commands_to_mcu_packets(commands: List[VibrationCommand],
                                fs_control: float = 50.0
                                ) -> List[Dict]:
        """
        Convert vibration commands to a list of time-stamped packets that can
        be sent from the smartphone to the patch MCU (BLE).

        This is just a reference data format; adapt it to your firmware protocol.

        Returns
        -------
        packets : list of dict
            Each dict contains: time, channel_mask, amplitude, frequency, mode
        """
        packets = []
        for cmd in commands:
            channel_mask = 0
            for ch in cmd.channels:
                channel_mask |= (1 << (ch - 1))
            packet = {
                "t_start": cmd.t_start,
                "t_end": cmd.t_end,
                "channel_mask": channel_mask,
                "amplitude": cmd.amplitude,
                "frequency": cmd.frequency,
                "mode": cmd.mode,
            }
            packets.append(packet)
        return packets
