"""Integration API registrations.

Registrations are intentionally best-effort so a missing optional robotics
stack (Franka, Agilex, R1Pro, vision dependencies, etc.) does not prevent other
robots from using their own lightweight APIs.
"""

from .base_api import list_apis, register_api


def _try_register(name: str, factory) -> None:
    try:
        register_api(name, factory)
    except Exception as exc:
        print(f"Skipping API {name}: {exc}")


try:
    from .r1lite.control import R1LiteControlApi

    _try_register("R1LiteControlApi", R1LiteControlApi)
except Exception as exc:
    print(f"R1Lite not installed, skipping R1Lite APIs: {exc}")


try:
    from .agilex.control import AgilexControlApi

    _try_register("AgilexControlApi", AgilexControlApi)
except Exception as exc:
    print(f"Agilex not installed, skipping Agilex APIs: {exc}")


try:
    from .r1pro.control import R1ProControlApi

    _try_register("R1ProControlApi", lambda env: R1ProControlApi(env, use_sam3=True))
except Exception as exc:
    print(f"R1Pro not installed, skipping R1Pro APIs: {exc}")


try:
    from .franka.control import FrankaControlApi
    from .franka.control_privileged import FrankaControlPrivilegedApi
    from .franka.control_reduced import FrankaControlApiReduced
    from .franka.control_reduced_exampleless import FrankaControlApiReducedExampleless
    from .franka.control_reduced_skill_library import (
        FrankaControlApiReducedSkillLibrary,
    )
    from .franka.handover import FrankaHandoverApi
    from .franka.handover_privileged import FrankaHandoverPrivilegedApi
    from .franka.handover_reduced import FrankaHandoverApiReduced
    from .franka.handover_reduced_exampleless import (
        FrankaHandoverApiReducedExampleless,
    )
    from .franka.nut_assembly_privileged import (
        FrankaControlNutAssemblyPrivilegedApi,
    )
    from .franka.nut_assembly_visual import FrankaControlNutAssemblyVisualApi
    from .franka.spill_wipe import FrankaControlSpillWipeApi
    from .franka.spill_wipe_privileged import FrankaControlSpillWipePrivilegedApi
    from .franka.two_arm_lift import FrankaTwoArmLiftApi
    from .franka.two_arm_lift_privileged import FrankaTwoArmLiftPrivilegedApi

    _try_register("FrankaControlPrivilegedApi", FrankaControlPrivilegedApi)
    _try_register("FrankaControlApi", lambda env: FrankaControlApi(env, use_sam3=True))
    _try_register("FrankaControlApiReduced", FrankaControlApiReduced)
    _try_register("FrankaControlApiReducedExampleless", FrankaControlApiReducedExampleless)
    _try_register("FrankaControlApiReducedSkillLibrary", FrankaControlApiReducedSkillLibrary)
    _try_register("FrankaHandoverApi", FrankaHandoverApi)
    _try_register("FrankaHandoverPrivilegedApi", FrankaHandoverPrivilegedApi)
    _try_register("FrankaHandoverApiReduced", FrankaHandoverApiReduced)
    _try_register("FrankaHandoverApiReducedExampleless", FrankaHandoverApiReducedExampleless)
    _try_register("FrankaControlNutAssemblyPrivilegedApi", FrankaControlNutAssemblyPrivilegedApi)
    _try_register("FrankaControlNutAssemblyVisualApi", FrankaControlNutAssemblyVisualApi)
    _try_register(
        "FrankaControlSpillWipeApi",
        lambda env: FrankaControlSpillWipeApi(
            env,
            tcp_offset=[0.0, 0.0, -0.0158],
            use_sam3=True,
        ),
    )
    _try_register("FrankaControlSpillWipePrivilegedApi", FrankaControlSpillWipePrivilegedApi)
    _try_register("FrankaTwoArmLiftApi", FrankaTwoArmLiftApi)
    _try_register("FrankaTwoArmLiftPrivilegedApi", FrankaTwoArmLiftPrivilegedApi)
    _try_register(
        "FrankaRealReducedSkillLibraryControlApi",
        lambda env: FrankaControlApiReducedSkillLibrary(
            env,
            tcp_offset=[0.0, 0.0, -0.157],
            real=True,
        ),
    )
    _try_register(
        "FrankaRealControlApi",
        lambda env: FrankaControlApi(
            env,
            tcp_offset=[0.0, 0.0, -0.157],
            real=True,
        ),
    )
except Exception as exc:
    print(f"Franka APIs not installed, skipping: {exc}")


try:
    from .franka.libero import FrankaLiberoApi
    from .franka.libero_privileged import FrankaLiberoPrivilegedApi
    from .franka.libero_reduced import FrankaLiberoApiReduced
    from .franka.libero_reduced_skill_library import (
        FrankaLiberoApiReducedSkillLibrary,
    )

    _try_register("FrankaLiberoPrivilegedApi", FrankaLiberoPrivilegedApi)
    _try_register("FrankaLiberoApi", lambda env: FrankaLiberoApi(env, use_sam3=True))
    _try_register("FrankaLiberoApiReduced", FrankaLiberoApiReduced)
    _try_register("FrankaLiberoApiReducedSkillLibrary", FrankaLiberoApiReducedSkillLibrary)
except Exception as exc:
    print(f"LIBERO APIs not installed, skipping: {exc}")
