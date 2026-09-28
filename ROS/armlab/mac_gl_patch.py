"""macOS GL fix: make robosuite's offscreen renderer use CGL instead of GLFW.

Why this is needed
------------------
On macOS, an interactive MuJoCo window requires `mjpython`, which runs the
Python interpreter on a *secondary* thread so the real main thread stays free
for Cocoa. But GLFW refuses to create anything (window OR hidden offscreen
context) off the main thread:

    NSInternalInconsistencyException: NSWindow should only be instantiated
    on the main thread!

robosuite hardcodes GLFW for the offscreen path on Darwin. MuJoCo ships a
thread-safe Apple CGL context with an identical constructor signature
(max_width, max_height), so we swap it in.

Effect: interactive viewer AND offscreen camera capture work in the same
mjpython process. Import this BEFORE importing robosuite.
"""

import platform
import sys
import types


def apply() -> bool:
    """Patch robosuite's GLFW context to CGL. Returns True if patched."""
    if platform.system() != "Darwin":
        return False

    if "robosuite" in sys.modules:
        raise RuntimeError("mac_gl_patch.apply() must run before importing robosuite")

    from mujoco.cgl import GLContext as CGLContext

    class CGLGLContext(CGLContext):
        """Drop-in replacement for robosuite's GLFWGLContext."""

        def __init__(self, max_width, max_height, device_id=0):
            super().__init__(max_width, max_height)

    # robosuite does `from robosuite.renderers.context.glfw_context import
    # GLFWGLContext as GLContext` lazily inside MjRenderContext.__init__, so
    # pre-seeding the module in sys.modules is enough - the real (GLFW-importing)
    # file is never touched.
    mod_name = "robosuite.renderers.context.glfw_context"
    stub = types.ModuleType(mod_name)
    stub.GLFWGLContext = CGLGLContext
    stub.__doc__ = "Patched by mac_gl_patch: CGL instead of GLFW."
    sys.modules[mod_name] = stub
    return True
