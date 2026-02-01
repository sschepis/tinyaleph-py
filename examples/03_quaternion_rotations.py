#!/usr/bin/env python3
"""
Example 03: Quaternion Rotations and Hypercomplex Numbers

This example demonstrates:
- Hamilton quaternion algebra (ℍ)
- 3D rotations using quaternions
- SLERP interpolation
- Hypercomplex numbers (sedenions)

Quaternions: q = w + xi + yj + zk where i² = j² = k² = ijk = -1

Key property: Quaternion multiplication is non-commutative!
    ij = k but ji = -k
"""

from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.hypercomplex import Hypercomplex
import math

def main():
    print("=" * 60)
    print("TinyAleph: Quaternion Algebra and Rotations")
    print("=" * 60)
    print()
    
    # ===== PART 1: Basic Quaternion Creation =====
    print("PART 1: Basic Quaternion Creation")
    print("-" * 40)
    
    # Create quaternion q = 1 + 2i + 3j + 4k
    q = Quaternion(1, 2, 3, 4)
    print(f"q = {q}")
    print(f"  Real part (w): {q.w}")
    print(f"  i component: {q.i}")
    print(f"  j component: {q.j}")
    print(f"  k component: {q.k}")
    print()
    
    # Special quaternions
    identity = Quaternion.identity()  # 1 + 0i + 0j + 0k
    zero = Quaternion.zero()          # 0 + 0i + 0j + 0k
    pure_i = Quaternion.basis_i()     # 0 + 1i + 0j + 0k
    pure_j = Quaternion.basis_j()     # 0 + 0i + 1j + 0k
    pure_k = Quaternion.basis_k()     # 0 + 0i + 0j + 1k
    
    print(f"Identity: {identity}")
    print(f"Basis i: {pure_i}")
    print(f"Basis j: {pure_j}")
    print(f"Basis k: {pure_k}")
    print()
    
    # ===== PART 2: Quaternion Arithmetic =====
    print("PART 2: Quaternion Arithmetic")
    print("-" * 40)
    
    q1 = Quaternion(1, 2, 0, 0)
    q2 = Quaternion(0, 0, 1, 1)
    
    print(f"q1 = {q1}")
    print(f"q2 = {q2}")
    print(f"q1 + q2 = {q1 + q2}")
    print(f"q1 - q2 = {q1 - q2}")
    print(f"2 * q1 = {q1 * 2}")
    print()
    
    # ===== PART 3: Hamilton Product =====
    print("PART 3: Hamilton Product (Non-Commutative!)")
    print("-" * 40)
    
    # The famous relations: i² = j² = k² = ijk = -1
    i = Quaternion.basis_i()
    j = Quaternion.basis_j()
    k = Quaternion.basis_k()
    
    print("Verifying fundamental relations:")
    print(f"  i² = {i * i} (expected: -1)")
    print(f"  j² = {j * j} (expected: -1)")
    print(f"  k² = {k * k} (expected: -1)")
    print()
    
    # ij = k, jk = i, ki = j
    print("Cyclic relations:")
    print(f"  ij = {i * j} (expected: k)")
    print(f"  jk = {j * k} (expected: i)")
    print(f"  ki = {k * i} (expected: j)")
    print()
    
    # Anti-commutative: ij ≠ ji
    print("Non-commutativity (ij ≠ ji):")
    print(f"  ij = {i * j}")
    print(f"  ji = {j * i}")
    print(f"  ij + ji = {i * j + j * i} (expected: 0)")
    print()
    
    # General product
    q1 = Quaternion(1, 2, 3, 4)
    q2 = Quaternion(5, 6, 7, 8)
    print(f"q1 * q2 = {q1 * q2}")
    print(f"q2 * q1 = {q2 * q1}")
    print(f"Are they equal? {(q1 * q2).w == (q2 * q1).w}")
    print()
    
    # ===== PART 4: Norm and Conjugate =====
    print("PART 4: Norm and Conjugate")
    print("-" * 40)
    
    q = Quaternion(1, 2, 3, 4)
    print(f"q = {q}")
    print(f"Norm ||q|| = {q.norm():.4f}")
    print(f"  (√(1² + 2² + 3² + 4²) = √30 = {math.sqrt(30):.4f})")
    print()
    
    # Conjugate: q* = w - xi - yj - zk
    conj = q.conjugate()
    print(f"Conjugate q* = {conj}")
    
    # q * q* = ||q||² (real number)
    product = q * conj
    print(f"q * q* = {product}")
    print(f"  (Should be ||q||² = {q.norm()**2:.4f})")
    print()
    
    # Normalization
    normalized = q.normalize()
    print(f"Normalized: {normalized}")
    print(f"Norm of normalized: {normalized.norm():.4f}")
    print()
    
    # Inverse: q⁻¹ = q* / ||q||²
    inv = q.inverse()
    print(f"Inverse q⁻¹ = {inv}")
    verify = q * inv
    print(f"q * q⁻¹ = {verify} (should be ≈ 1)")
    print()
    
    # ===== PART 5: Axis-Angle Representation =====
    print("PART 5: Axis-Angle Representation for Rotations")
    print("-" * 40)
    
    # A unit quaternion represents a 3D rotation:
    # q = cos(θ/2) + sin(θ/2)(ax*i + ay*j + az*k)
    # where (ax, ay, az) is the rotation axis and θ is the angle
    
    # Rotation of 90° (π/2) around z-axis
    angle = math.pi / 2
    axis = (0, 0, 1)  # z-axis
    
    rot_z_90 = Quaternion.from_axis_angle(axis, angle)
    print(f"90° rotation around z-axis: {rot_z_90}")
    print(f"  cos(45°) = {math.cos(angle/2):.4f}")
    print(f"  sin(45°) = {math.sin(angle/2):.4f}")
    print()
    
    # 180° rotation around x-axis
    rot_x_180 = Quaternion.from_axis_angle((1, 0, 0), math.pi)
    print(f"180° rotation around x-axis: {rot_x_180}")
    print()
    
    # ===== PART 6: Rotating Vectors =====
    print("PART 6: Rotating 3D Vectors")
    print("-" * 40)
    
    # To rotate vector v by quaternion q: v' = q * v * q⁻¹
    # (where v is treated as a pure quaternion 0 + v_x*i + v_y*j + v_z*k)
    
    # Rotate (1, 0, 0) by 90° around z-axis
    # Should give (0, 1, 0)
    v = (1, 0, 0)
    rotated = rot_z_90.rotate_vector(v)
    print(f"Rotate {v} by 90° around z-axis:")
    print(f"  Result: ({rotated[0]:.4f}, {rotated[1]:.4f}, {rotated[2]:.4f})")
    print(f"  Expected: (0, 1, 0)")
    print()
    
    # Rotate (0, 1, 0) by 180° around x-axis
    # Should give (0, -1, 0)
    v = (0, 1, 0)
    rotated = rot_x_180.rotate_vector(v)
    print(f"Rotate {v} by 180° around x-axis:")
    print(f"  Result: ({rotated[0]:.4f}, {rotated[1]:.4f}, {rotated[2]:.4f})")
    print(f"  Expected: (0, -1, 0)")
    print()
    
    # ===== PART 7: Composing Rotations =====
    print("PART 7: Composing Rotations")
    print("-" * 40)
    
    # Quaternion multiplication composes rotations!
    # q2 * q1 rotates first by q1, then by q2
    
    rot_x_90 = Quaternion.from_axis_angle((1, 0, 0), math.pi / 2)
    rot_y_90 = Quaternion.from_axis_angle((0, 1, 0), math.pi / 2)
    
    # First rotate around x, then around y
    combined = rot_y_90 * rot_x_90
    print(f"90° around X then 90° around Y: {combined}")
    
    # Apply to (1, 0, 0)
    v = (1, 0, 0)
    result = combined.rotate_vector(v)
    print(f"  Rotating {v}: ({result[0]:.4f}, {result[1]:.4f}, {result[2]:.4f})")
    print()
    
    # ===== PART 8: SLERP Interpolation =====
    print("PART 8: SLERP (Spherical Linear Interpolation)")
    print("-" * 40)
    
    # SLERP smoothly interpolates between two rotations
    # slerp(q1, q2, t) for t ∈ [0, 1]
    
    q_start = Quaternion.identity()
    q_end = Quaternion.from_axis_angle((0, 0, 1), math.pi)  # 180° around z
    
    print("Interpolating from identity to 180° z-rotation:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = q_start.slerp(q_end, t)
        # Extract rotation angle
        angle = 2 * math.acos(min(1.0, max(-1.0, q_interp.w)))
        print(f"  t={t:.2f}: angle = {math.degrees(angle):.1f}°")
    print()
    
    # ===== PART 9: Euler Angles =====
    print("PART 9: Euler Angles Conversion")
    print("-" * 40)
    
    # Convert between quaternions and Euler angles (roll, pitch, yaw)
    
    q = Quaternion.from_euler(
        roll=math.radians(30),
        pitch=math.radians(45),
        yaw=math.radians(60)
    )
    print(f"From Euler (30°, 45°, 60°): {q}")
    
    roll, pitch, yaw = q.to_euler()
    print(f"Back to Euler:")
    print(f"  Roll: {math.degrees(roll):.1f}°")
    print(f"  Pitch: {math.degrees(pitch):.1f}°")
    print(f"  Yaw: {math.degrees(yaw):.1f}°")
    print()
    
    # ===== PART 10: Exponential and Logarithm =====
    print("PART 10: Exponential and Logarithm")
    print("-" * 40)
    
    # exp(q) and log(q) for quaternions
    # Useful for smooth interpolation and Lie group operations
    
    # For unit quaternions: q = exp(θ/2 * (ax*i + ay*j + az*k))
    # log(q) = θ/2 * (ax*i + ay*j + az*k)
    
    q = Quaternion.from_axis_angle((0, 0, 1), math.pi / 3)  # 60° around z
    print(f"Original: {q}")
    
    log_q = q.log()
    print(f"log(q) = {log_q}")
    
    exp_log_q = log_q.exp()
    print(f"exp(log(q)) = {exp_log_q}")
    print(f"  (Should equal original)")
    print()
    
    # ===== PART 11: Hypercomplex Numbers =====
    print("PART 11: Hypercomplex Numbers (Sedenions)")
    print("-" * 40)
    
    # Sedenions are 16-dimensional hypercomplex numbers
    # Used for holographic memory encoding
    
    import numpy as np
    
    # Create a sedenion
    sed = Hypercomplex(16)  # 16-dimensional
    print(f"Zero sedenion: {sed}")
    print(f"  Dimension: {sed.dim}")
    
    # Create from components
    components = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=float)
    sed = Hypercomplex(16, components)
    print(f"Sedenion with components 1-16:")
    print(f"  Norm: {sed.norm():.4f}")
    print(f"  Entropy: {sed.entropy():.4f}")
    print()
    
    # Conjugate
    conj = sed.conjugate()
    print(f"First 4 components of conjugate: {conj.c[:4]}")
    print("  (Real part preserved, imaginary parts negated)")
    print()
    
    # Scalar multiplication
    scaled = sed * 2.0
    print(f"After scaling by 2: norm = {scaled.norm():.4f}")
    print()
    
    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Quaternions and Hypercomplex Numbers")
    print("=" * 60)
    print("""
Quaternion Algebra (ℍ):
- q = w + xi + yj + zk
- i² = j² = k² = ijk = -1
- Non-commutative: ij ≠ ji
- Unit quaternions represent 3D rotations

Key Operations:
- Norm: ||q|| = √(w² + x² + y² + z²)
- Conjugate: q* = w - xi - yj - zk
- Inverse: q⁻¹ = q* / ||q||²
- Hamilton product: (q1 * q2)

Rotation Properties:
- Rotation by θ around axis (a): q = cos(θ/2) + sin(θ/2)(a·i,j,k)
- Rotate vector v: v' = q * v * q⁻¹
- Compose rotations: q_combined = q2 * q1 (right-to-left)
- SLERP: Smooth interpolation between rotations

Hypercomplex Numbers:
- Sedenions: 16-dimensional (for holographic memory)
- Extend quaternion structure to higher dimensions
- Used in SMF (Sedenion Memory Field)
    """)

if __name__ == "__main__":
    main()