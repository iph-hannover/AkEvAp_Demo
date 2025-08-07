import time
import numpy as np
from enum import Enum
import utils


class CouplingType(Enum):
    GOOD = 0
    FAIR = 1
    POOR = 2


class NIOSH:

    def __init__(self, skeleton_type, load_weight, initial_frame_count, coupling_type: CouplingType = CouplingType.FAIR):
        self.skeleton_type = skeleton_type
        self.V_start = None
        self.t_start = None
        self.t_lifts = []
        self.load_weight = load_weight
        self.worker_height = 1.65  # assume worker of average height (165 cm) for vertical component computation
        self.coupling_type = coupling_type

        self.frame_count = initial_frame_count
        self.fps = 30

    def stop_task(self):
        self.t_start = None

    def get_time(self):
        # as we do not have real-time capable GUI yet, we need to compute time deltas on frame basis
        return self.frame_count / self.fps
        #return time.time()

    def start_lifting(self, points3d):
        load_center = utils.compute_load_center(points3d, self.skeleton_type)

        # remember for distance and frequency multiplier
        self.V_start = load_center[1] * 100  # in cm

        # remember for frequency multiplier
        if self.t_start is None:
            self.t_start = self.get_time()

        # compute multipliers
        HM = self.get_horizontal_multiplier(load_center)
        VM = self.get_vertical_multiplier(load_center)
        DM = dict(value=None, info={'vertical travel distance (D)': 'n/A'})  #  cannot be computed at the beginning of a lift
        AM = self.get_asymmetric_multiplier(load_center)
        FM = self.get_frequency_multiplier(load_center)
        CM = self.get_coupling_multiplier(load_center)
        multipliers = dict(HM=HM, VM=VM, DM=DM, AM=AM, FM=FM, CM=CM)

        # remember for frequency multiplier
        self.t_lifts.append(self.get_time())  # remember start time of this lift for frequency computation
        while True:
            lift = self.t_lifts[0]
            if self.get_time() - lift < 15 * 60:
                break
            self.t_lifts.pop(0)  # only count lifts within last 15 minutes => drop older ones

        # compute resulting scores
        RWL = self.get_recommended_weight_limit(multipliers)
        LI = self.get_lifting_index(self.load_weight, RWL)
        scores = dict(RWL=RWL, LI=LI)

        return multipliers, scores

    def compute_after_lift(self, points3d):
        ### LIMITATIONS OF NIOSH ###
        # - not designed to assess tasks involving one-handed lifting, lifting while seated or kneeling
        # - assumed that no pivoting or stepping occurs. Although this assumption may overestimate the reduction in acceptable load weight, it will provide the greatest protection for the worker
        #   => equation will still apply if there is a small amount of holding and carrying, but carrying should be limited to one or two steps and holding should not exceed a few seconds

        load_center = utils.compute_load_center(points3d, self.skeleton_type)

        # compute multipliers
        HM = self.get_horizontal_multiplier(load_center)
        VM = self.get_vertical_multiplier(load_center)
        DM = self.get_distance_multiplier(load_center)
        AM = self.get_asymmetric_multiplier(load_center)
        FM = self.get_frequency_multiplier(load_center)
        CM = self.get_coupling_multiplier(load_center)
        multipliers = dict(HM=HM, VM=VM, DM=DM, AM=AM, FM=FM, CM=CM)

        # compute resulting scores
        RWL = self.get_recommended_weight_limit(multipliers)
        LI = self.get_lifting_index(self.load_weight, RWL)
        scores = dict(RWL=RWL, LI=LI)

        # reset state
        self.V_start = None

        return multipliers, scores

    def get_horizontal_multiplier(self, load_center):
        # horizontal location (H): mid-point of the line joining the inner ankle bones to a point projected on the floor directly below the mid-point of the hand grasps (i.e., load center)
        #       If significant control is required at the destination (i.e., precision placement), then H should be measured at both the origin and destination of the lift.
        #       If the horizontal distance is less than 10 inches (25 cm), then H is set to 10 inches (25 cm) [most objects that are closer than this cannot be lifted without encountering interference from the abdomen or hyperextending the shoulders]
        #       25 inches (63 cm) was chosen as the maximum value for H [cannot be lifted vertically without some loss of balance]
        # Horizontal Multiplier (HM): 25/H for H measured in centimeters
        #       If H is less than or equal to 10 inches (25 cm), then the multiplier is 1.0
        #       The multiplier for H is reduced to 0.4 when H is 25 inches (63 cm). If H is greater than 25 inches, then HM=0.
        #       => check HM using Table 1
        load_center_proj = np.asarray([load_center[0], 0, load_center[2]])
        H = np.linalg.norm(load_center_proj) * 100  # in cm
        HM = 25/max(H, 25)
        HM = HM if H < 63 else 0
        return dict(value=HM, info={'horizontal location (H)': (H, 'cm')})

    def get_vertical_multiplier(self, load_center):
        # vertical location (V): vertical height of the hands above the floor
        #       The vertical location should be measured at the origin and the destination of the lift to determine the travel distance (D)
        # Vertical Multiplier (VM): absolute value or deviation of V from an optimum height of 30 inches (75 cm)
        #       height of 30 inches above floor level is considered “knuckle height” for a worker of average height (66 inches or 165 cm) # TODO: scale thresholds by body height?
        #       VM = (1−(.003 [V−75]) for V measured in centimeters
        #               => When V is at 30 inches (75 cm), the vertical multiplier (VM) is 1.0.
        #               => At floor level VM is 0.78, and at 70 inches (175 cm) height VM is 0.7
        #       If V is greater than 70 inches: VM=0
        #       => check VM using Table 2
        V = load_center[1] * 100  # in cm
        VM = 1 - .003 * np.abs(V - 75)
        VM = VM if V < 175 else 0
        return dict(value=VM, info={'vertical location (V)': (V, 'cm')})

    def get_distance_multiplier(self, load_center):
        # vertical travel distance (D): defined as the vertical travel distance of the hands between the origin and destination of the lift
        #       For lifting, D can be computed by subtracting the vertical location (V) at the origin of the lift from the corresponding V at
        #       the destination of the lift (i.e., D is equal to V at the destination minus V at the origin).
        #       For a lowering task, D is equal to V at the origin minus V at the destination.
        #       The variable (D) is assumed to be at least 10 inches (25cm), and no greater than 70 inches [175cm].
        #       If the vertical travel distance is less than 10 inches (25 cm), then D should be set to the minimum distance of 10 inches (25 cm).
        # Distance Multiplier (DM):
        #       .82+(4.5/D)
        #       DM ranges from 1.0 to 0.85 as the D varies from 0 inches (0 cm) to 70 inches (175 cm).
        #       => check DM using Table 3
        V = load_center[1] * 100  # in cm
        D = np.abs(self.V_start - V)
        DM = .82 + 4.5 / max(D, 25)
        DM = DM if D < 175 else 0
        return dict(value=DM, info={'vertical travel distance (D)': (D, 'cm')})

    def get_asymmetric_multiplier(self, load_center):
        # asymmetry angle (A): defined as the angle between the asymmetry line and the mid-sagittal line
        # asymmetry line: horizontal line that joins the mid-point between the inner ankle bones and the point projected on the floor directly below the mid-point of the hand grasps
        # sagittal line: line passing through the mid-point between the inner ankle bones and lying in the mid-sagittal plane
        # mid-sagittal plane: neutral body position (i.e., hands directly in front of the body, with no twisting at the legs, torso, or shoulders)
        # neutral body position: position of the body when the hands are directly in front of the body and there is minimal twisting at the legs, torso, or shoulders
        # Asymmetric Multiplier (AM):
        #       AM = 1−(.0032A)
        #       maximum value of 1.0 when the load is lifted directly in front of the body
        #       Range: value of 0.57 at 135° of asymmetry to a value of 1.0 at 0° of asymmetry (i.e., symmetric lift)
        #       If A is greater than 135°, then AM = 0, and the load is zero.
        #       => check AM using Table 4
        load_center_proj = np.asarray([load_center[0], 0, load_center[2]]) * 100  # in cm
        A = np.arctan2(load_center_proj[0], load_center_proj[2]) * 180 / np.pi  # arctan2(y, x) computes angle regarding (x=1, y=0)
        if load_center_proj[2] < 20:
            print("WARNING: hands are less than 20cm in front of body -> angle might be too noisy")
            print("         => might happen due to imprecise start/end frame of lifting (e.g. in a walking frame)")
            A = 0
        AM = 1 - .0032 * np.abs(A)
        AM = AM if A < 135 else 0
        return dict(value=AM, info={'asymmetry angle (A)': (A, 'deg')})

    def get_frequency_multiplier(self, load_center):
        # The frequency multiplier is defined by
        #       (a) the number of lifts per minute (frequency),
        #       (b) the amount of time engaged in the lifting activity (duration), and
        #       (c) the vertical height of the lift from the floor.
        # Lifting frequency (F):
        #       average number of lifts made per minute, as measured over a 15-minute period
        # Lifting duration: classified into three categories
        #       Short-duration: work duration <= 1 hour, followed by a recovery time equal to 1.0 times the work time
        #           => at least a 1.0 recovery-time to work-time ratio (RT/WT)
        #       Moderate-duration: duration > 1 hour AND <= 2 hours, followed by a recovery period of at least 0.3 times the work time
        #           => at least a 0.3 recovery-time to work-time ratio (RT/WT)
        #       Long-duration: duration between 2-8 hours, with standard industrial rest allowances (e.g., lunch breaks)
        # Frequency Restrictions:
        #       maximum frequency that is dependent on the vertical location of the object (V) and the duration of lifting
        # Frequency Multiplier (FM): depends upon
        #       - the average number of lifts/min (F),
        #       - the vertical location (V) of the hands at the origin and
        #       - the duration of continuous lifting
        # => check VM using Table 5

        lifting_span = (self.get_time() - self.t_lifts[0]) / 60 if len(self.t_lifts) else 0  # timespan in minutes in which lifting was performed

        use_special_procedure = lifting_span < 13  # simply always use special procedure when sampling period is too short (TODO: check for continuous lifting)
        if use_special_procedure:
            # use Special Frequency Adjustment Procedure to compensate discontinuous lifting (lifting_span < 15min)
            # => When using this special procedure, the duration category is based on the magnitude
            #    of the recovery periods between work sessions, not within work sessions.
            # => intermittent recovery periods that occur during the 15-minute sampling period are not
            #    considered as recovery periods for purposes of determining the duration category
            # 1. Compute the total number of lifts performed for the 15 minute period
            # 2. Divide the total number of lifts by 15
            F = len(self.t_lifts) / 15
            #assert len(self.t_lifts) / (lifting_span+1e-8) < 15  # using special procedure is only valid as long as the actual lifting frequency does not exceed 15 lifts per minute
        else:
            F = len(self.t_lifts) / (lifting_span+1e-8)  # lifting frequency in [lifts/min]
        V = self.V_start / 2.54  # here we need V in [inches]
        WD = (self.get_time() - self.t_start) / 60 / 60  # overall work duration in [hours]
        # TODO: consider recovery-time -> start/end of lifting session/breaks as user input or recommendations?
        #  => avoid necessity to track work time (WT) and recovery time (RT) as RT is known retrospectively only

        LUT = {
            0.2: [1.00, 1.00, .95, .95, .85, .85],
            0.5: [.97, .97, .92, .92, .81, .81],
            1: [.94, .94, .88, .88, .75, .75],
            2: [.91, .91, .84, .84, .65, .65],
            3: [.88, .88, .79, .79, .55, .55],
            4: [.84, .84, .72, .72, .45, .45],
            5: [.80, .80, .60, .60, .35, .35],
            6: [.75, .75, .50, .50, .27, .27],
            7: [.70, .70, .42, .42, .22, .22],
            8: [.60, .60, .35, .35, .18, .18],
            9: [.52, .52, .30, .30, .00, .15],
            10: [.45, .45, .26, .26, .00, .13],
            11: [.41, .41, .00, .23, .00, .00],
            12: [.37, .37, .00, .21, .00, .00],
            13: [.00, .34, .00, .00, .00, .00],
            14: [.00, .31, .00, .00, .00, .00],
            15: [.00, .28, .00, .00, .00, .00],
            np.inf: [.00, .00, .00, .00, .00, .00],
        }
        freqs = np.asarray(list(LUT.keys()))
        freq = freqs[freqs >= F].min()
        WD_idx = 0 if WD <= 1 else 1 if WD <= 2 else 2
        FM = LUT[freq][WD_idx*2 + (V >= 30)]
        return dict(value=FM, info={'lifting frequency (F)': (F, 'lifts/min'), 'work duration (WD)': (WD, 'hours'), 'vertical location (V_start) of the hands at the origin': (self.V_start, 'cm')})

    def get_coupling_multiplier(self, load_center):
        # A good coupling will reduce the maximum grasp forces required and increase the acceptable weight for lifting,
        # while a poor coupling will generally require higher maximum grasp forces and decrease the acceptable weight for lifting
        #       - The effectiveness of the coupling is not static, but may vary with the distance of the object from the ground
        #       - The entire range of the lift should be considered when classifying hand-to-object couplings, with classification based on overall effectiveness
        #       - If there is any doubt about classifying a particular coupling design, the more stressful classification should be selected.
        #       => check CM using Table 6 & 7 + Decision Tree for Coupling Quality
        V = load_center[1] * 100  # in cm
        LUT = {
            CouplingType.GOOD: [1.0, 1.0],
            CouplingType.FAIR: [.95, 1.0],
            CouplingType.POOR: [.90, .90],
        }
        CM = LUT[self.coupling_type][V >= 75]
        return dict(value=CM, info={'coupling type (CT)': f"{self.coupling_type.name} [{self.coupling_type.value}]", 'vertical location (V)': (V, 'cm')})

    def get_recommended_weight_limit(self, multipliers):
        # Recommended Weight limit (RWL):
        #       defined for a specific set of task conditions as the weight of the load that nearly all healthy workers
        #       could perform over a substantial period of time (e.g., up to 8 hours) without an increased risk of developing lifting-related LBP
        LC = 23  # load constant: 23 kg
        RWL = LC
        for multiplier in multipliers.values():
            RWL = RWL * (multiplier['value'] if multiplier['value'] is not None else 1.0)
        return RWL

    def get_lifting_index(self, load_weight, RWL):
        # Lifting Index (LI):
        #       provides a relative estimate of the level of physical stress associated with a particular manual lifting task
        #       The estimate of the level of physical stress is defined by the relationship of the weight of the load lifted and the recommended weight limit
        LI = load_weight / RWL
        return LI
