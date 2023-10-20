uniform float v_time;

void main()
{
	float fR = 0.9 * sin(0.0 + v_time*0.05) + 1.0;
	float fG = 0.9 * cos(0.33 + v_time*0.05) + 1.0;
	float fB = 0.9 * sin(0.67 + v_time*0.05) + 1.0;
	gl_FragColor = vec4(fR/2.0, fG/2.0, fB/2.0, 1.0);
}

