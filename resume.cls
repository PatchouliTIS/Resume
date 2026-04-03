\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{resume}[2015/05/25 An elegant Résumé class]
\LoadClass[11pt]{article}

% disable indent globally
\setlength{\parindent}{0pt}
% some general improvements, defines the XeTeX logo
\RequirePackage{xltxtra}
% use xifthen
\RequirePackage{xifthen}
% use fontawesome
\RequirePackage{fontawesome}
% use xcolor for customizing color
\RequirePackage[usenames,dvipsnames]{xcolor}
% loading fonts
\RequirePackage{fontspec}
% Main document font
\setmainfont[
  Path = fonts/Main/ ,
  Extension = .otf ,
  UprightFont = *-regular ,
  BoldFont = *-bold ,
  ItalicFont = *-italic ,
  BoldItalicFont = *-bolditalic ,
  SmallCapsFont = Fontin-SmallCaps
]{texgyretermes}

\RequirePackage[
	paperwidth=210mm,
	paperheight=999mm,
	left=0.7in,
	right=0.7in,
	top=0.50in,
	bottom=0.5in,
	nohead
]{geometry}

% Auto-fit page height to content (single page output)
% Use two-pass approach: write content height to .aux, read it on next compile
\newlength{\savedContentHeight}
\setlength{\savedContentHeight}{0pt}

% Read saved height from previous compile (if available)
\AtBeginDocument{%
  \ifdefined\resumeContentHeight
    \setlength{\savedContentHeight}{\resumeContentHeight}%
    \addtolength{\savedContentHeight}{1.0in}% top + bottom margins
    \newgeometry{
      paperwidth=210mm,
      paperheight=\savedContentHeight,
      left=0.7in,
      right=0.7in,
      top=0.50in,
      bottom=0.5in,
      nohead
    }%
    \pdfpageheight=\savedContentHeight
  \fi
}

% Save current content height to .aux for next compile
\AtEndDocument{%
  \immediate\write\@auxout{%
    \string\gdef\string\resumeContentHeight{\the\pagetotal}%
  }%
}

\RequirePackage{titlesec}
\RequirePackage{enumitem}
\setlist{noitemsep} % removes spacing from items but leaves space around the whole list
%\setlist{nosep} % removes all vertical spacing within and around the list
\setlist[itemize]{topsep=0.25em, leftmargin=1.5pc}
\setlist[enumerate]{topsep=0.25em, leftmargin=1.5pc}
\RequirePackage[super]{nth}

\titleformat{\section}         % Customise the \section command 
  {\Large\scshape\raggedright} % Make the \section headers large (\Large),
                               % small capitals (\scshape) and left aligned (\raggedright)
  {}{0em}                      % Can be used to give a prefix to all sections, like 'Section ...'
  {}                           % Can be used to insert code before the heading
  [\titlerule]                 % Inserts a horizontal line after the heading
\titlespacing*{\section}{0cm}{*1}{*1}

\titleformat{\subsection}
  {\large\raggedright}
  {}{0em}
  {}
\titlespacing*{\subsection}{0cm}{*1}{*0.5}

\newcommand{\datedsection}[2]{%
  \section[#1]{#1 \hfill #2}%
}
\newcommand{\datedsubsection}[2]{%
  \subsection[#1]{#1 \hfill #2}%
}
\newcommand{\datedline}[2]{%
  {\par #1 \hfill #2 \par}%
}

\newcommand{\name}[1]{
  \centerline{\Huge\scshape{#1}}
  \vspace{1.2ex}
}

\newcommand{\contactInfo}[4]{
  \centerline{\sffamily\large{\ {#1} \textperiodcentered\ \ {#2}}
    \ifthenelse{\isempty{#3}}%
      { } % if {#3} is empty
      {\textperiodcentered\ \ {#3} } % homepage, no space before \text...
    \ifthenelse{\isempty{#4}}%
      { } % if {#3} is empty
      {\textperiodcentered\ \ {#4} }
  }
  \vspace{0.7ex}
}

\newcommand{\role}[2]{
  {\par \textit{#1} ~ #2 \par}
  \vspace{0.5ex}
}
