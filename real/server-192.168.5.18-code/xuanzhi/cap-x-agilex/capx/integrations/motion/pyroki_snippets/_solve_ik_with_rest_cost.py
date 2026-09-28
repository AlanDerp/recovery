"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_rest(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    rest_cost_weights: jax.Array | float=0.0,
    initial_q: onp.ndarray | None = None,
    soft_limit_margin: float = 0.0,
    soft_limit_weight: float = 0.0,
    joint_lower: onp.ndarray | None = None,
    joint_upper: onp.ndarray | None = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    if initial_q is None:
        initial_q = onp.array(robot.joint_var_cls.default_factory())
    initial_q = onp.asarray(initial_q, dtype=onp.float64).reshape(
        robot.joints.num_actuated_joints
    )
    if joint_lower is None:
        joint_lower = onp.full(robot.joints.num_actuated_joints, -onp.inf)
    if joint_upper is None:
        joint_upper = onp.full(robot.joints.num_actuated_joints, onp.inf)
    cfg = _solve_ik_jax_rest(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        rest_cost_weights,
        initial_q,
        jnp.array(float(max(0.0, soft_limit_margin))),
        jnp.array(float(max(0.0, soft_limit_weight))),
        jnp.asarray(joint_lower),
        jnp.asarray(joint_upper),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax_rest(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    rest_cost_weights: jax.Array,
    initial_q: jax.Array,
    soft_limit_margin: jax.Array,
    soft_limit_weight: jax.Array,
    joint_lower: jax.Array,
    joint_upper: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]

    def soft_limit_residual(vals: jaxls.VarValues, var: jaxls.Var[jnp.ndarray]) -> jax.Array:
        q = vals[var]
        margin = jnp.maximum(soft_limit_margin, 1e-6)
        lower_violation = jnp.maximum((joint_lower + margin) - q, 0.0)
        upper_violation = jnp.maximum(q - (joint_upper - margin), 0.0)
        return ((lower_violation + upper_violation) / margin * soft_limit_weight).flatten()

    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=initial_q,
            weight=rest_cost_weights,
        ),
        jaxls.Cost(
            soft_limit_residual,
            (joint_var,),
            name="soft_limit_residual",
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(initial_q)]
            ),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]
