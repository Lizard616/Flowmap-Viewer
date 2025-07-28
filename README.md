# Flowmap-Viewer
A PyOpenGL-based interactive viewer that maps animated 2D flow textures onto a 3D sphere; ideal for simulating gas giant atmospheres, ocean currents, or energy fields.

 Download
 Download the EXE from the Releases tab

Features
Animated flowmap preview on a gas giant-style sphere

Smooth orbit camera (yaw & pitch centered)

Adjustable flow speed

Seamless looping animation

Standalone .exe – no install needed

Controls
Left Mouse Drag: Orbit around the sphere

Settings Tab: Change flow speed

Buttons: Load your own Diffuse Map and Flow Map (.png, .jpg, .bmp)

# Note
No Python installation required – this app runs as-is.
Just download, extract, and launch the .exe.

# Bugs
There are two main issues with the Flowmap Viewer. The camera is not properly centered when moving side-to-side around the sphere, resulting in uneven and wonky horizontal movement. The vertical movement works correctly, but the camera drifts off-center during horizontal rotation. Additionally, the flow animation on the sphere does not loop seamlessly. There is a visible jump or flicker at the loop point that disrupts the smooth flow effect.

Steps to Reproduce:
Open the Flowmap Viewer with the default sphere and flowmap. Move the camera side-to-side and observe that it is not centered, causing awkward movement. Then, watch the flow animation and notice the jump or flicker when it loops.

Expected Behavior:
The camera should remain perfectly centered on the sphere during all rotations, ensuring smooth and consistent movement. The flow animation should loop seamlessly with no visible jumps or flickering.
