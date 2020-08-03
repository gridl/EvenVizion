## N/A 
coordinates
The 
coordinates 
can’t 
be   defined
 when 
the 
matching points 
are clinging
 to 
moving 
objects. 
This
 means 
that 
the 
filtration 
isn’t 
working 
well 
enough. 
The 
coordinates 
can’t 
be   defined
 also 
when 
camera 
rotation
 angle 
is   more 
than 
90°. 
As   a solution 
to   the 
first 
problem we 
now 
consider 
applying
video 
motion 
segmentation 
to distinguish
 static 
points 
from 
motion 
points 
(taking 
into 
consideration 
the 
nature 
of movement). 
As   a solution 
to the 
second 
problem 
we   see 
the 
transfer
 to the 
cylindrical 
coordinate 
system.

## H=None
To   find
 the homography 
you 
need 
at least
 4   matching
 points. 
But 
in   some 
cases 
the 
4  points 
can’t 
be   found, 
and 
the 
homography 
matrices are
Н=None.
In 
a  current 
algorithm
 version 
we   process 
such 
cases 
this 
way:
 if 
the 
argument 
none_H_processing
 is   set 
for 
True 
we consider 
the 
matrix 
of the 
previous 
frame 
matches 
the 
matrix 
for 
the 
current 
frame 
(Hk=Hk-1). 
If set 
for 
False,
 then 
H=[[1,0,0][0,1,0][0,0,1]], 
meaning 
that 
there 
were 
no 
movement 
in   the 
frame.
 It is necessary 
to   think 
over 
better 
handling of 
such 
situations.
 
## Error
There’s 
an inaccuracy 
in   the 
coordinates. 
Poorly 
obtained
 homography
matrix 
distorts 
the 
results 
of   coordinate 
recalculation. The 
reasons for 
that 
are:
Poor 
filtration.
 If the 
points 
catch 
on   a  motion 
object, then
 the 
homography 
matrix 
will 
describe
 not 
only the
 camera 
movement,
 but 
also 
the 
independent 
movement
 of   objects 
(for 
example,
 a  person's walking).
Using 
the 
built-in 
findHomography 
()   function
 of   the OpenCV 
module. 
This 
function 
already
 assumes 
there 
is   an   error 
in the calculation 
of the 
