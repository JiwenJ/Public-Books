/*
 * @brief Passthrough vertex shader
 * @author Deyuan Qiu
 * @date May 8, 2009
 * @file passthrough.vert
 */

void main()
{
//	the following three lines provide the same result
//	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
//	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_Position = ftransform();
}
