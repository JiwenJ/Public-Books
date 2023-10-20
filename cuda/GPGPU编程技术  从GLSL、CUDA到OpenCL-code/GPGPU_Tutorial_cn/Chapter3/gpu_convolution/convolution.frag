#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect texture;
uniform float fRadius;
float nWidth = 3.0;
float nHeight = 3.0;

void main(void) {
	//get the current texture location
	vec2 pos = gl_TexCoord[0].st;

	vec4 fSum = vec4(0.0, 0.0, 0.0, 0.0);		//Sum of the neighborhood.
	vec4 fTotal = vec4(0.0, 0.0, 0.0, 0.0);		//NoPoints in the neighborhood.
	vec4 vec4Result = vec4(0.0, 0.0, 0.0, 0.0);	//Output vector to replace the current texture.

	//Neighborhood summation.
	for (float ii = pos.x - fRadius; ii < pos.x + fRadius + 0.5; ii += 1.0)	//plus 1.0 for the '0.5 effect'.
		for (float jj = pos.y - fRadius; jj <= pos.y + fRadius + 0.5; jj += 1.0) {
			if (ii >= 0.0 && jj >= 0.0 && ii < nWidth && jj < nHeight) {
				fSum += texture2DRect(texture, vec2(ii, jj));
				fTotal += vec4(1.0, 1.0, 1.0, 1.0);
			}
		}
	vec4Result = fSum / fTotal;

	gl_FragColor = vec4Result;
}
