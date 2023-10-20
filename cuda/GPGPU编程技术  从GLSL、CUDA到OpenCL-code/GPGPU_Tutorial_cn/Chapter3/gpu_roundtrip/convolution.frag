uniform sampler2D texScene, texModel;
uniform float fWidth, fHeight, fThres;
uniform int nRadius;

void main(void)
{
   float fStepX = 1.0/fWidth;
   float fStepY = 1.0/fHeight;
   vec2 v2Offset = vec2(0.0,0.0);
   float fDist = 0.0;
   float fDistMin = 100000.0;
   float fR=0.0;
   float fG=1.0;
   float fB=0.0;
   float fA=0.0;
   int i=0;
   int j=0;

   for(i=-1*nRadius;i<nRadius+1;i++)
   {
	   for(j=-1*nRadius;j<nRadius+1;j++)
	   {
		   v2Offset.r = gl_TexCoord[0].s+float(i)*fStepX;
		   v2Offset.g = gl_TexCoord[0].t+float(j)*fStepY;
		   if((v2Offset.r>=0.0)&&(v2Offset.r<=1.0)&&(v2Offset.g>=0.0)&&(v2Offset.g<=1.0))
		   {
			   fDist=distance(texture2D(texScene, gl_TexCoord[0].st).rgb,texture2D(texModel, v2Offset).rgb);
			   if(fDist<fDistMin)
			   {
				   fDistMin=fDist;
				   fR=fDist;											//minimum distance by now
				   fG=texture2D(texModel, v2Offset)[3];					//model point index
				   fB=texture2D(texScene, gl_TexCoord[0].st)[3];		//scene point index
			   }
		   }
	   }
   }

   if(fR>fThres)														//cut outliers
   {
	   fG=0.0;
	   fA=1.0;
   }

   gl_FragColor = vec4(fR,fG,fB,fA);
}
