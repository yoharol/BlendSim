from manim import *

class BezierCurveScene(Scene):
    def construct(self):

        self.camera.background_color = WHITE
        Text.set_default(color=BLACK, font="Times New Roman")
        MathTex.set_default(color=BLACK)
        Dot.set_default(color=BLUE)
        
        

        # Step 1: Show three points and label them
        q_j_minus1 = np.array([-4, -0.7, 0])
        q_j = np.array([0, 2, 0])
        q_j_plus1 = np.array([4, -0.7, 0])

        dot_q_j_minus1 = Dot(q_j_minus1)
        dot_q_j = Dot(q_j)
        dot_q_j_plus1 = Dot(q_j_plus1)

        label_q_j_minus1 = MathTex(r'\mathbf{x}_{i-1}').next_to(dot_q_j_minus1, DOWN)
        label_q_j = MathTex(r'\mathbf{x}_{i}').next_to(dot_q_j, UP)
        label_q_j_plus1 = MathTex(r'\mathbf{x}_{i+1}').next_to(dot_q_j_plus1, DOWN)

        self.play(FadeIn(dot_q_j_minus1), FadeIn(dot_q_j), FadeIn(dot_q_j_plus1))
        self.play(Write(label_q_j_minus1), Write(label_q_j), Write(label_q_j_plus1))
        self.wait(2.0)


        # ======================== show linear interpolation ========================

        title1 = Text("Linear Interpolation", font_size=300).scale(0.1).to_edge(UL)
        line_1 = Line(q_j_minus1, q_j, color=BLUE)
        line_2 = Line(q_j, q_j_plus1, color=BLUE)

        self.play(Create(title1))
        self.play(Create(line_1), Create(line_2))

        moving_point1 = Dot(color=RED).scale(1.6)
        t_tracker = ValueTracker(0)

        def update_point(mob):
            # Get the current value of t from the tracker
            t = t_tracker.get_value()
            # Calculate the point on the Bezier curve at parameter t
            if t < 1:
                point_on_curve = interpolate(q_j_minus1, q_j, t)
                mob.move_to(point_on_curve)
            else:
                t = t - 1
                point_on_curve = interpolate(q_j, q_j_plus1, t)
                mob.move_to(point_on_curve)
        
        moving_point1.add_updater(update_point)
        self.add(moving_point1)
        self.play(t_tracker.animate.set_value(2), run_time=3, rate_func=linear)

        self.play(FadeOut(moving_point1), FadeOut(line_1), FadeOut(line_2), FadeOut(title1))
        self.remove(moving_point1)
        self.wait(1.0)

        # ======================== show traditional method ========================

        # Step 2: Generate arrows at each point, pointing in a direction
        v_j_minus1 = np.array([1, 2, 0])
        v_j = np.array([2, 0, 0])
        v_j_plus1 = np.array([1, -2, 0])

        arrow_scale = 1.5

        arrow_v_j_minus1 = Arrow(
            start=q_j_minus1, 
            end=q_j_minus1 + (v_j_minus1 / np.linalg.norm(v_j_minus1)) * arrow_scale, 
            buff=0,
            color=BLUE
        )

        arrow_v_j = Arrow(
            start=q_j, 
            end=q_j + (v_j / np.linalg.norm(v_j)) * arrow_scale, 
            buff=0,
            color=BLUE
        )

        arrow_v_j_plus1 = Arrow(
            start=q_j_plus1, 
            end=q_j_plus1 + (v_j_plus1 / np.linalg.norm(v_j_plus1)) * arrow_scale, 
            buff=0,
            color=BLUE
        )

        d1 = np.linalg.norm(q_j - q_j_minus1)
        d2 = np.linalg.norm(q_j_plus1 - q_j)

        h1 = d1 / 3
        h2 = d2 / 3

        # First Bezier segment control points
        P0 = q_j_minus1
        P1 = q_j_minus1 + (v_j_minus1 / np.linalg.norm(v_j_minus1)) * h1
        P2 = q_j - (v_j / np.linalg.norm(v_j)) * h1
        P3 = q_j

        # Second Bezier segment control points
        Q0 = q_j
        Q1 = q_j + (v_j / np.linalg.norm(v_j)) * h2
        Q2 = q_j_plus1 - (v_j_plus1 / np.linalg.norm(v_j_plus1)) * h2
        Q3 = q_j_plus1

        # Create Bezier curves
        bezier1 = CubicBezier(P0, P1, P2, P3, color=RED)
        bezier2 = CubicBezier(Q0, Q1, Q2, Q3, color=RED)

        v0_j_minus1 = np.array([1, 0, 0])
        v0_j = np.array([1, 0, 0])
        v0_j_plus1 = np.array([1, 0, 0])

        arrow_v0_j_minus1 = Arrow(
            start=q_j_minus1,
            end=q_j_minus1 + (v0_j_minus1 / np.linalg.norm(v0_j_minus1)) * arrow_scale,
            buff=0,
            color=BLUE
        )

        arrow_v0_j = Arrow(
            start=q_j,
            end=q_j + (v0_j / np.linalg.norm(v0_j)) * arrow_scale,
            buff=0,
            color=BLUE
        )

        arrow_v0_j_plus1 = Arrow(
            start=q_j_plus1,
            end=q_j_plus1 + (v0_j_plus1 / np.linalg.norm(v0_j_plus1)) * arrow_scale,
            buff=0,
            color=BLUE
        )

        # Step 3: Label the arrows
        label_v0_j_minus1 = MathTex(r'\mathbf{v}_{i-1}').next_to(arrow_v0_j_minus1.get_end(), UP)
        label_v0_j = MathTex(r'\mathbf{v}_{i}').next_to(arrow_v0_j.get_end(), UP)
        label_v0_j_plus1 = MathTex(r'\mathbf{v}_{i+1}').next_to(arrow_v0_j_plus1.get_end(), UP)


        # First Bezier segment control points
        P00 = q_j_minus1
        P01 = q_j_minus1 + (v0_j_minus1 / np.linalg.norm(v0_j_minus1)) * h1
        P02 = q_j - (v0_j / np.linalg.norm(v0_j)) * h1
        P03 = q_j

        # Second Bezier segment control points
        Q00 = q_j
        Q01 = q_j + (v0_j / np.linalg.norm(v0_j)) * h2
        Q02 = q_j_plus1 - (v0_j_plus1 / np.linalg.norm(v0_j_plus1)) * h2
        Q03 = q_j_plus1

        # Create Bezier curves
        bezier01 = CubicBezier(P00, P01, P02, P03, color=RED)
        bezier02 = CubicBezier(Q00, Q01, Q02, Q03, color=RED)

        linear_points = []
        bezier_points = []

        # Sampling points on the linear segments
        for alpha in np.linspace(0, 1, 10):
            point_on_line_1 = line_1.point_from_proportion(alpha)
            point_on_line_2 = line_2.point_from_proportion(alpha)
            linear_points.append(Dot(point_on_line_1, color=ORANGE))
            linear_points.append(Dot(point_on_line_2, color=ORANGE))

        # Sampling 10 points on the Bezier curve
        for alpha in np.linspace(0, 1, 10):
            point_on_bezier_1 = bezier1.point_from_proportion(alpha)  # For first half of the curve
            point_on_bezier_2 = bezier2.point_from_proportion(alpha)  # For second half
            bezier_points.append(point_on_bezier_1)
            bezier_points.append(point_on_bezier_2)
        
        traditional_text = Text("Traditional Method", font_size=300).scale(0.1).to_edge(UL)
        self.play(Create(traditional_text))
        self.play(*[FadeIn(dot) for dot in linear_points])

        left_point = np.array([-4.8, 2.8, 0])
        right_point = np.array([-1.3, 2.8, 0])
        optimize_points = []
        for alpha in np.linspace(0, 1, 20):
            t = alpha
            point_on_line = left_point + t * (right_point - left_point)
            optimize_points.append(Dot(point_on_line, color=ORANGE))

        
        optimize_text = Text("Optimize:", font_size=300).scale(0.1).next_to(traditional_text, DOWN, aligned_edge=LEFT)
        self.play(Create(optimize_text) )
        self.play(*[FadeIn(dot) for dot in optimize_points])

        move_animations = [
            dot.animate.move_to(bezier_points[i]) for i, dot in enumerate(linear_points)
        ]
        self.play(*move_animations, run_time=3)
        heavy_text = Text("Heavy computation", font_size=200, color=RED).scale(0.1).next_to(optimize_text, DOWN, aligned_edge=LEFT)
        # export_text = Text("Export as frame-by-frame animation", font_size=20, color=RED).next_to(heavy_text, DOWN, aligned_edge=LEFT)
        incompatible_text = Text("Difficult to edit", font_size=200, color=RED).scale(0.1).next_to(heavy_text, DOWN, aligned_edge=LEFT)
        self.play(Create(heavy_text))
        # self.play(Create(export_text))
        self.play(Create(incompatible_text))
        self.wait(2)

        self.play(FadeOut(optimize_text), FadeOut(traditional_text), FadeOut(heavy_text), FadeOut(incompatible_text),  *[FadeOut(dot) for dot in linear_points], *[FadeOut(dot) for dot in optimize_points])

        left_point = np.array([-4.8, 2.8, 0])
        right_point = np.array([-3.8, 2.8, 0])
        optimize_points = []
        for alpha in np.linspace(0, 1, 3):
            t = alpha
            point_on_line = left_point + t * (right_point - left_point)
            optimize_points.append(Dot(point_on_line, color=ORANGE))

        our_method_text = Text("BlendSim", font_size=300).scale(0.1).to_edge(UL)
        self.play(Write(our_method_text))

        self.play(GrowArrow(arrow_v0_j_minus1), GrowArrow(arrow_v0_j), GrowArrow(arrow_v0_j_plus1))
        self.play(Write(label_v0_j_minus1), Write(label_v0_j), Write(label_v0_j_plus1))

        self.play(Create(bezier01))
        self.play(Create(bezier02))
        self.wait(2.0)

        # Optional: Show control points and lines (uncomment to display)
        cp_dots1 = [Dot(p, color=ORANGE) for p in [P00, P01, P03]]
        cp_lines1 = VGroup(
            Line(P00, P01, color=ORANGE),
        )

        cp_dots2 = [Dot(p, color=ORANGE) for p in [Q01, Q02, Q03]]
        cp_lines2 = VGroup(
            Line(Q00, Q01, color=ORANGE),
            Line(Q02, Q03, color=ORANGE)
        )

        self.play(FadeOut(label_v0_j), FadeOut(label_v0_j_minus1), FadeOut(label_v0_j_plus1), FadeOut(arrow_v0_j), FadeOut(arrow_v0_j_minus1), FadeOut(arrow_v0_j_plus1), *[FadeIn(dot) for dot in cp_dots1 + cp_dots2])
        self.play(Create(cp_lines1), Create(cp_lines2))
        self.play(Create(optimize_text))
        self.play(*[FadeIn(dot) for dot in optimize_points])
        self.wait(0.5)


        self.play(
            cp_dots1[1].animate.move_to(P1),
            cp_dots2[0].animate.move_to(Q1),
            cp_dots2[1].animate.move_to(Q2),
            cp_lines1[0].animate.put_start_and_end_on(P0, P1),
            cp_lines2[0].animate.put_start_and_end_on(Q0, Q1),
            cp_lines2[1].animate.put_start_and_end_on(Q2, Q3),
            Transform(bezier01, bezier1),
            Transform(bezier02, bezier2),
            run_time=3.0
        )

        moving_point2 = Dot(color=RED).scale(1.6)
        t_tracker = ValueTracker(0)

        def update_point(mob):
            # Get the current value of t from the tracker
            t = t_tracker.get_value()
            # Calculate the point on the Bezier curve at parameter t
            if(t < 1):
                point_on_curve = bezier1.point_from_proportion(t)
                mob.move_to(point_on_curve)
            else:
                t = t - 1
                point_on_curve = bezier2.point_from_proportion(t)
                mob.move_to(point_on_curve)
        
        moving_point2.add_updater(update_point)
        self.add(moving_point2)
        self.play(t_tracker.animate.set_value(2), run_time=3, rate_func=linear)

        fast_text = Text("Fast computation", font_size=200, color=GREEN).scale(0.1).next_to(optimize_text, DOWN, aligned_edge=LEFT)
        interactive_text = Text("Interactive editing", font_size=200, color=GREEN).scale(0.1).next_to(fast_text, DOWN, aligned_edge=LEFT)
        export_text = Text("Export as blendshape animation", font_size=200, color=GREEN).scale(0.1).next_to(interactive_text, DOWN, aligned_edge=LEFT)
        # compatible_text = Text("Compatible with animation software", font_size=20, color=GREEN).next_to(export_text, DOWN, aligned_edge=LEFT)

        self.play(Create(fast_text))
        self.play(Create(interactive_text))
        self.play(Create(export_text))
        # self.play(Create(compatible_text))
        self.wait(2)

