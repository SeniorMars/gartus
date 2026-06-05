General Notes:

Items seperated by | means you must choose one of them in an input line.
Items enclosed in [] are optional.

For example, rotate is specified as:
rotate x|y|z degress [knob]

The following would be valid rotations:
rotate x 20
rotate y 23 k1

While the following would be invalid:
rotate x|y 20
rotate x y 33
rotate x 33 [k1]

Stack Commands
--------------
push	- makes a new top level of stack and COPIES the previous top
	  to the new top level. REMEMBER, THIS MUST BE A COPY, NOT
	  A POINTER TO THE OLD TOP

pop	- pops off the top of the stack (doesn't return anything)


Transformations
---------------
All transformations will operate as follows:

1. If there is a knob, scale the transformation by the knob value.
2. Create the appropriate transformation matrix M
3. Multiply top_of_stack * M and replace the old top of the stack
   with this result.

move x y z [knob]		- translate
scale x y z [knob]		- scale
rotate x|y|z degrees [knob]	- rotate (note that you can only
				  specify one axis, x, y, or z per
				  rotation instruction)

Image creation
--------------
All image creation commands will operate as follows:

1. Generate all the points and edges for the object in question.
2. If no coord_system is specified, transform the points against the
   top of the stack. If there is a coord_system specified, transform 
   the points against that matrix instead.
3. Render the object.
4. Throw away the point list (if this is applicable in your implementation).

sphere [constants] x y z r [coord_system]

torus [constants] x y z r0 r1  [coord_system]

box [constants] x0 y0 z0 h w d [coord_system]
			- x0 y0 z0 = one corner of the box
			- h w d = height width and depth

line [constants] x0 y0 z0 [coord_system0] x1 y1 z1 [coord_system1]
			- NOTE: each endpoint of the line can be drawn
			  in its own coordinate system.

mesh [constants] :filename [coord_system]
			- load a mesh or set of edges (in some format that
			  you can specify) from a file into the pointlist 
			  and or edge list directly.

Knobs/Animation
---------------
basename name		- sets the base filename to save under.
			  This should be used in conjunction with
			  either vary or tween. For example,
			  if "name" is R, then tween or vary might
			  save the images r01.miff, r02.miff etc.

set knobname value	- sets a knobs value (in the symbol table).

save_knobs knoblist	- saves the current values of all knobs
			  under the name "knoblist."
    
tween start_frame end_frame knoblist0 knoblist1
			- generates a number of frames using basename
			  as the base filename. It will start from
			  start_frame and end at end_frame and
			  interpolate the image using knoblist0 as
			  the starting configuration and knoblist 2
			  as the ending configuration.

frames num_frames	- How many frames to generate all together.

vary knob start_frame end_frame start_val end_val
			- vary a knob from start_val to end_val over
			  the course of start_frame to end_frame
setknobs value		- set all the knobs to value


Lighting
--------
light r g b x y z  	- creates a "light" datastructure with rgb values
			  r,g,b at location x,y,z.
			  This is inserted into the symbol table.

ambient r g b 		- specifies how much ambient light is in the scene

constants name kar kdr ksr kag kdg ksg kab kdb ksb [r] [g] [b]
			- saves a set of lighting components in the
			  symbol table under "name."
 			- r g b intensities can be specified. If not specified, they 
			  default to 0.

shading wireframe|flat|gouraud|phong|raytrace
			- set the shading mode


MISC
----
//			- comment to the end of a line, just like c++
save_coord_system name
			- Makes a copy of the top of the stack and 
			  saves it in the symbol table under "name."

camera eye aim		- establishes a camera. Eye and aim are
			  x y z triples.


save filename		- save the image in its current state under
			  the name "filename."

gereate_rayfiles	- Instruct the interpreter to generate source
			  files for a ray tracer for each frame rendered.

focal value		- set the focal length of the camera

display			- display the current image on the screen


LEX & BISON NOTES
-----------------
main() is defined in mdl.y. It is defined as follows:

	int main() {
  	yyparse();
		print_pcode();
	}

    print_pcode() merely prints the results of the parse.
    You will probably want to remove that call from your final
    system. The routine yyparse() runs the parser. It reads the script
    and fills the symbol table and the global command arry op[].

    Rather than add your own code in main, I would recommend making a
    new driver routine (mymain(), for example) and put it in another
    file. After calling yyparse(), call your routine.

    NOTE: You can access the symbol table and op array as long as you include
	  symtab.h and parser.h in your files.

There are only two lines that are really two commands in the Makefile
that need explaining.

The first is:	flex -I mdl.l

This command takes the file describing the lexical analyzer, mdl.l,
and builds the analyzer (which will return tokens one at a time from
the script file). Flex creates a file lex.yy.c which will then be
compiled into our project.

The second is: 	bison -d -y mdl.y

This command takes the parser description in mdl.y and builds the
parser.  It creates two files, y.tab.c and y.tab.h. These contain a
number of things including #defines, a routine called yyparse() which
does the actual parse, and in our case, main(). The option "-y" tells
bison to use yacc's naming convensions (bison is gnus implementation
of yacc and it has a number of extensions) specifically that the
output files are called y.tab.h and y.tab.c. The "-d" option tells
bison to create y.tab.h (in addition to y.tab.c) which contains the
#defines that are used in both the lexical analyzer and in our code.
