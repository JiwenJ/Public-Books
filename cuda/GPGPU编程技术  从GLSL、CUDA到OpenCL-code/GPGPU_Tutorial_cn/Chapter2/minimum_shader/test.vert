void main(){
	vec4 a;
	a = gl_ModelViewProjectionMatrix * gl_Vertex;
	gl_Position.x = 0.4 * a.x;
	gl_Position.y = 0.1 * a.y;
}
