# SMLP2021: Advanced methods in frequentist statistics with Julia 
Notebooks for "Advanced methods in frequentist statistics with Julia" as part of [SMLP2021](https://vasishth.github.io/smlp2021/)

## Instructors

| Name            | GitHub                                   |
|-----------------|------------------------------------------|
| Phillip Alday   | [@palday](https://github.com/palday)     |
| Douglas Bates   | [@dmbates](https://github.com/dmbates)    |
| Lisa DeBruine   | [@debruine](https://github.com/debruine) |
| Reinhold Kliegl | [@kliegl](https://github.com/kliegl)     |

## Schedule

The summer school takes place from Monday, 6 September to Friday, 10 September.  
We will have default time slots with content filled depending on assessment of progress by instructors and acutely expressed demands by participants.
The stream will probably split into two sub-streams for the final two or three days, possibly also already on the first two days.
The time schedule will respect time slots for SMLP2021-wide events (e.g., keynotes). With these constraints, we have four time slots online sessions per day: 

Approximate daily blocks/timeslots:

| Block | Time of Day | Approximate Time |
|:-----:|:------------|-----------------:|
| 1     | Early Morning | 9:00 - 10:20 |
| Break | Coffee  | 10:20 - 10:40 |
| 2.    | Late Morning | 10:40 - 12.00 |
| Break | Lunch | 12:00 - 14:00 |
| 3.    | Early Afternoon |  14:00 - 15:10 | 
| Break | Coffee  | 15:10 - 15:30 |
| 4.    | Late Afternoon | 15:30 - 17:00 |   

**NB: Precise timing may vary slightly depending on topic and participants' needs.** 

We will be meeting via [Zoom](https://uni-potsdam.zoom.us/j/64170432269). For obvious reasons, we aren't posting the passcode here; please ask in the associated stream chat.

Slots are filled with one of five "modules" dealing with  

|Module| Topic |
|:--:|:-------------------------------------------|
| M1 |  IDE workflow  (VS Code, Pluto notebooks)  |
| M2 | Julia programming language (minimal) |
| M3 | Targeted presentations (ranging from LMM introductions on the first days  to very specialized LMM topics, e.g., power simulation. on later days) |
| M4 | Working with participants’ data and scripts |
| M5 | Open question and answer sessions (a format accidentally discovered last year that worked very well) |

Monday and Tuesday, we will largely focus on M1-3, alternating so as not to overload any topic. As the week progresses, we'll swap more and more to M3-M5.

## Software

Please install:
- [Julia 1.6.2 (the current stable release)](https://julialang.org/downloads/)
- [VSCode](https://code.visualstudio.com/) and the Julia extension (extensions can be installed from within VSCode).

The documentation for the Julia VSCode includes a [useful installation guide](https://www.julia-vscode.org/docs/dev/gettingstarted/#Installation-and-Configuration-1) for doing all of this. 

After you have installed Julia, we recommend that you also try to install a Julia package to verify your set up. For our purposes, the MixedModels.jl package is very apt. When you start Julia, you immediately see the REPL ("read-evaluate-print loop", i.e. the Julia command prompt):

```julia
julia>
```

If you type `]` at the REPL prompt, the REPL changes *modes* to the package manager mode
```julia
(@v1.6) pkg> 
```

From here, you can execute package management commands, such as `add MixedModels` to install MixedModels.jl. We recommend also installing a few other packages to speed a few things along later:

```
(@v1.6) pkg> add CairoMakie DataFrames MixedModels Pluto # note that there are no commas in package mode
```

To get back to the normal REPL mode, simply hit backspace at the `pkg>` prompt. 
If you type `?`, you'll swap to the helpmode, where you can access documentation:
```julia
help?> sqrt
search: sqrt isqrt

  sqrt(x)

  Return \sqrt{x}. Throws DomainError for negative Real arguments. Use complex
  negative arguments instead. The prefix operator √ is equivalent to sqrt.

  Examples
  ≡≡≡≡≡≡≡≡≡≡
...
```
Access to the shell REPL mode (i.e. the default terminal) is achieved by `;` at the `julia>` prompt. 
There are also other useful REPL modes provided by various Julia packages, such as the R mode provided by the Julia package `RCall` (more on that during the course).

We will start the summer school with chats in the zoomApp probably in late August to get the technical installation problems out of the way before the first meeting. We will send installation notes around as soon as the version of the Julia programming language and the version of MixedModels.jl that we want to use for the summer school are released; also of VS Code. Currently, there are new versions in the pipeline. Of course, you don’t have to wait for our instructions, especially if you already feel comfortable with updating.

## Code and Notebooks

**NB: Notebooks begin executing their code, including potentially heavy computations, as soon as they are opened in Pluto.**

The code and notebooks we use in teaching this course (and potentially some we don't get a chance to use) are in this repository on GitHub. If you're comfortable using git, great! If you're not, then using a graphical interface (such as the one provided by [GitHub](https://desktop.github.com/) or [SourceTree](https://www.sourcetreeapp.com/)) will make your life easier at first. We really recommend becoming comfortable with at least the basics of version control with git + GitHub (or GitLab, if that's your or your institute's preferred hosting website). Version control makes reproducible science a lot easier, and it is also a convenient mechanism for us, the instructors to continuously add to and revise the course materials based on your feedback and immediately distribute those changes to you.

You can also view a statically rendered version of the notebooks [here](https://repsychling.github.io/SMLP2021/). Each notebook also contains a link in the upper left hand corner showing you how to run the notebook on your own computer. Note that normal Julia scripts, i.e. code that we don't put into notebooks, will not be available through this route.

If you're starting Pluto from your local Julia REPL (make sure that you've `add`ed it as discussed above), you'll see something like this:

```julia
               _         
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |                                                                       
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |                                                                       
  | | |_| | | | (_| |  |  Version 1.6.2 (2021-07-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |                    
                                               
julia> using Pluto; Pluto.run()             
                                               
Go to http://localhost:1234/?secret=YaQMOJKJ in your browser to start writing ~ have fun!
                                               
Press Ctrl+C in this terminal to stop Pluto 

```

On most platforms, your browser should automatically a tab with the Pluto.
