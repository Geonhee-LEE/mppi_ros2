"""
GPU 가속 MPPI 모듈

PyTorch CUDA 백엔드를 사용한 MPPI 파이프라인 가속.
device="cuda" 설정 시 자동으로 GPU 경로 활성화.
"""

from mppi_controller.controllers.mppi.gpu.torch_dynamics import (
    TorchDiffDriveKinematic,
    TorchDynamicsWrapper,
)
from mppi_controller.controllers.mppi.gpu.torch_costs import TorchCompositeCost
from mppi_controller.controllers.mppi.gpu.torch_sampling import TorchGaussianSampler
from mppi_controller.controllers.mppi.gpu.torch_models import (
    TorchAckermannKinematic,
    TorchAckermannDynamic,
    TorchSwerveDriveKinematic,
    TorchSwerveDriveDynamic,
)
from mppi_controller.controllers.mppi.gpu.torch_learned import TorchNeuralDynamics


def get_torch_model(model, device="cuda"):
    """
    RobotModel에 대응하는 Torch 모델 반환

    Args:
        model: RobotModel 인스턴스
        device: torch device 문자열

    Returns:
        Corresponding Torch GPU model
    """
    from mppi_controller.models.kinematic.differential_drive_kinematic import (
        DifferentialDriveKinematic,
    )
    from mppi_controller.models.kinematic.ackermann_kinematic import (
        AckermannKinematic,
    )
    from mppi_controller.models.dynamic.ackermann_dynamic import (
        AckermannDynamic,
    )
    from mppi_controller.models.kinematic.swerve_drive_kinematic import (
        SwerveDriveKinematic,
    )
    from mppi_controller.models.dynamic.swerve_drive_dynamic import (
        SwerveDriveDynamic,
    )
    from mppi_controller.models.learned.neural_dynamics import NeuralDynamics

    # NeuralDynamics → TorchNeuralDynamics (nn.Module 직접 감싸기)
    if isinstance(model, NeuralDynamics):
        return TorchNeuralDynamics(model, device=device)

    if isinstance(model, DifferentialDriveKinematic):
        return TorchDiffDriveKinematic(device=device)

    if isinstance(model, AckermannKinematic):
        return TorchAckermannKinematic(
            wheelbase=model.wheelbase, device=device
        )

    if isinstance(model, AckermannDynamic):
        return TorchAckermannDynamic(
            wheelbase=model.wheelbase, c_v=model.c_v, device=device
        )

    if isinstance(model, SwerveDriveKinematic):
        return TorchSwerveDriveKinematic(device=device)

    if isinstance(model, SwerveDriveDynamic):
        return TorchSwerveDriveDynamic(
            c_v=model.c_v, c_omega=model.c_omega, device=device
        )

    raise ValueError(
        f"GPU acceleration not supported for {model.__class__.__name__}. "
        f"Supported: DifferentialDriveKinematic, AckermannKinematic, "
        f"AckermannDynamic, SwerveDriveKinematic, SwerveDriveDynamic, NeuralDynamics"
    )


__all__ = [
    "TorchDiffDriveKinematic",
    "TorchDynamicsWrapper",
    "TorchCompositeCost",
    "TorchGaussianSampler",
    "TorchAckermannKinematic",
    "TorchAckermannDynamic",
    "TorchSwerveDriveKinematic",
    "TorchSwerveDriveDynamic",
    "TorchNeuralDynamics",
    "get_torch_model",
]
