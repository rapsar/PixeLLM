# ```PixeLLM```: Outpainting with LLMs
LLMs are (among other things) pattern-matching models.
Given a grayscale image with pixel values 0-255, they can 'understand' and extrapolate an image passed as:
```
0,0,0,255;
0,0,255,0;
0,255,0,0;
```
and continue it with:
```
255,0,0,0;
```

Use ```draw.py``` to draw or upload an image, pass it in string format to an LLM, and see the extrapolation.


