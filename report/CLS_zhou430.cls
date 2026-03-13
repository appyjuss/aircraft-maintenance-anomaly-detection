
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CLS_zhou430}[2024/12/26 Custom class for CE 430 Senior Project]

% Base class
\LoadClass[12pt]{report}

% Packages
\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{hyperref}
\RequirePackage{setspace}

\usepackage{listings}
\usepackage{xcolor}

\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue}\bfseries,
    stringstyle=\color{red},
    commentstyle=\color{gray},
    numbers=left,
    numberstyle=\tiny,
    stepnumber=1,
    frame=single,
    breaklines=true,
    captionpos=b,
    escapeinside={(*@}{@*)}
}

% Custom Commands
\newcommand{\coursenumber}{CE 430}
\newcommand{\coursetitle}{Senior Project}
\newcommand{\supervisor}{Dr. Zhaoxian Zhou}
\newcommand{\department}{School of Computing Sciences and Computer Engineering}
\newcommand{\institution}{University of Southern Mississippi}
\newcommand{\dateofsubmission}{\today}
\newcommand{\deadlineofsubmission}{05/08/2026}


% Page Layout
\setlength{\textwidth}{6.5in}
\setlength{\textheight}{9in}
\setlength{\oddsidemargin}{0in}
\setlength{\evensidemargin}{0in}
\setlength{\topmargin}{-0.5in}
\setlength{\headheight}{0in}
\setlength{\headsep}{0.2in}
\setlength{\footskip}{0.4in}



% Define a Counter for Authors
\newcounter{authorcounter}

% Initialize the Author List Macro
\newcommand{\@authorlist}{} % Initialize as empty

% Command for Multiple Authors with Name, ID, and Affiliation
\newcommand{\addauthor}[3]{
    \stepcounter{authorcounter}
    \g@addto@macro\@authorlist{
        \textbf{#1}  % Name
        (\textit{#2}),  % ID
        #3 \\[1em] % Affiliation
    }
}

% Author Metadata Environment
\newenvironment{authormetadata}{
    \begin{center}
	Written by \\     \vspace{2em}
     \large \@authorlist
    \end{center}
    \vspace{2em}
}



% Title Page
\newcommand{\maketitlepage}{
% Title Page
\makeatletter
\begin{titlepage}
    \centering
    { \includegraphics[width=4.5 in]{figs/fig_usm.png}}\\[0.2in]
    {\LARGE \textbf{\projecttitle}}\\[0.2in]
    Presented to \\[.1in]
    
    \supervisor \\[0.3in]
    
    In Partial Fulfillment of the Requirements for \\ [.1in]
    \coursenumber{} \coursetitle  \\[0.4in]
    
    \begin{authormetadata}
    \end{authormetadata}

    {\department }\\[0.3in]
    {\institution} \\[0.4in]
    \date{\large \textbf{Date Last Edited: \today}}
    {\@date\\} 
    \vspace{0.2in}
    \date{\large \textit{Deadline of Submission: \deadlineofsubmission}}
    {\@date\\}

\end{titlepage}
\makeatother}



% Scoring sheet
\newcommand{\makescoringpage}{
\begin{table}[ht]
    \centering
    \textbf{\Large Scoring Sheet Based on Preset Rubrics} \\[0.3in]

    \begin{tabular}{|p{0.7\textwidth}|p{0.3\textwidth}|} \hline
         \textbf{Section} & \textbf{Scoring} \\ \hline 
         \rule{0pt}{6ex} 0: Abstract (5\%)& \\ \hline  
         \rule{0pt}{6ex} 1: Introduction (5\%)& \\ \hline 
         \rule{0pt}{6ex} 2: Review (5\%)& \\ \hline 
         \rule{0pt}{6ex} 3: Engineering Standards (5\%)& \\ \hline 
         \rule{0pt}{6ex} 4: Methodology (15\%)& \\ \hline 
         \rule{0pt}{6ex} 5: Design Contraints (5\%)& \\ \hline 
         \rule{0pt}{6ex} 6: Design/Imlementation (20\%)& \\ \hline 
         \rule{0pt}{6ex} 7: Results/Evaluation (20\%)& \\ \hline 
         \rule{0pt}{6ex} 8: Discussion/Conclusion (5\%)& \\ \hline 
         \rule{0pt}{6ex} 9: References (5\%)& \\ \hline 
         \rule{0pt}{6ex} 10: Writing (5\%)& \\ \hline 
         \rule{0pt}{6ex} 11: Originality (5\%)& \\ \hline 
         \rule{0pt}{6ex} 12: Bonus (5\%)& \\ \hline 
         \rule{0pt}{6ex} OVERALL (100\%)& \\ \hline 
    \end{tabular}
%    \caption{Grades}
%    \label{tab:my_label}
\end{table}
\newpage
}


% front matter
% Frontmatter Command
\newcommand{\fmname}{frontmatter} % Customizable abstract title
\newenvironment{frontmatter}{
\maketitlepage
\makescoringpage
\tableofcontents
}



% abstract
% Abstract Command
\renewcommand{\abstractname}{Abstract} % Customizable abstract title
\renewenvironment{abstract}{
    \newpage
	\chapter*{Abstract}
	\addcontentsline{toc}{chapter}{Abstract}
}{
    \vspace{2em}
}


% Acknowledgments
% Acknowledgments Command
\newcommand{\acknname}{Acknowledgments} % Customizable abstract title
\newenvironment{acknowledgments}{
    \newpage
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments}
\noindent
}{
    \vspace{2em}
}



% Lists
\newcommand{\makelists}{
% Lists of Figures and Tables
\listoffigures
\addcontentsline{toc}{chapter}{List of Figures}
\noindent
\listoftables
\addcontentsline{toc}{chapter}{List of Tables}
\noindent
}





% End of class definition
