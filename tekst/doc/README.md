The KU Leuven Engineering Master's Thesis Class
===============================================
The kulemt LaTeX class helps you to format your Master's thesis according to
the rules of the KU Leuven Faculty of Engineering Science. It can be adapted
to other Master's thesis layouts by tweaking a configuration file.

Bug reports and remarks can be sent to
[Luc Van Eycken](http://www.kuleuven.be/wieiswie/en/person/00010701).


Table of contents
-----------------
* Documentation files
* Installation instructions
* Change history ("what's new?")


Documentation files
-------------------
  * `guidelines_thesis.pdf`: guidelines for writing a Master's thesis at the
    KU Leuven Faculty of Engineering Science
  * `kulemt.pdf`: user manual of kulemt (KU Leuven Engineering Master Thesis
    document class)
  * `kulemt-src.pdf`: documented source (of `kulemt.cls` and `kulemt.cfg`)
  * `kulemtx.pdf`: user manual and documented source of the package kulemtx
    (kulemt extension)
  * `sjabloon`: directory with Dutch templates
  * `template`: directory with English templates

All documentation is available from the `kulemt-doc.zip` file or from the
`doc/latex/kulemt` directory in the `kulemt-tds.zip` file. To locate it on your
system after installation, read the installation instructions below.


Installation instructions
-------------------------

### Quick and dirty installation on any system ###
This is not the preferred way to install this class, because it only works
if all the LaTeX files of your thesis are in a single directory, which we
shall call `$XYZ`, and the documentation must be installed separately.
However, if everything else fails or you want to put `$XYZ` on Overleaf,
this is the way to go.

 1) Before installing this package, make sure that the memoir class is
    installed on your TeX system. Check out the documentation of your TeX
    installation to find out how to do this.

 2) Unzip `kulemt-tex.zip` (from the
    [ftp site](ftp://ftp.esat.kuleuven.be/latex/kulemt/)) into the
    `$XYZ` directory. 

Optionally, you can install the documentation where ever you want.

 3) Unzip `kulemt-doc.zip` (from the ftp site) into your preferred
    documentation directory. 


### Proper installation on a Unix like system ###
A Unix like system is any recent TeX installation on a Unix machine, MacOS X,
or Cygwin on Windows. These TeX installations are all based on TeXLive.

 1) Before installing this package, make sure that the memoir class is
    installed on your TeX system. To check this, issue the command
   
        kpsewhich memoir.cls
   
    and check if a file path is echoed.

 2) Find a suitable texmf tree to install in. We'll use `$ROOT` to refer to
    the root directory of that tree.

      * For a system-wide installation:
   
            ROOT=`kpsewhich -expand-var='$TEXMFLOCAL'`

      * For an installation for the current user only:
      
            ROOT=`kpsewhich -expand-var='$TEXMFHOME'`
	  
        Note: some older installations use HOMETEXMF instead of TEXMFHOME.
        If no directory name is returned, use the `texmf` subdirectory of
	    your home directory:
	  
            ROOT="$HOME/texmf"
   
    Make sure this root directory exists:
   
        mkdir -p "$ROOT"

 3) Unzip `kulemt-tds.zip` (from the
    [ftp site](ftp://ftp.esat.kuleuven.be/latex/kulemt/)) in `$ROOT`:
   
        unzip -d "$ROOT" kulemt-tds.zip

 4) If you also want to install the sources, unzip `kulemt-src.zip` (from
    the ftp site) in the appropriate directory:
   
        mkdir -p "$ROOT/source/latex"
        unzip -d "$ROOT/source/latex" kulemt-src.zip

 5) Update the filename database (only needed for system-wide installations):

        mktexlsr "$ROOT"

Note: All documentation is available in the `$ROOT/doc/latex/kulemt` directory.


### Proper installation on MikTeX ###
This also includes MikTeX derived installations such as proTeXt.

Note: The following procedure has only been tested on MikTeX 2.8.
      The MikTeX options can be accessed through the start menu
      MikTeX 2.8 | Maintenance | Settings. For system-wide installations,
      use "Maintenance (Admin)" instead of "Maintenance".

 1) Before installing this package, make sure that you enabled the automatic
    installation of missing packages ("Package installation" on the General
    tab of the MikTeX options)

 2) Find a suitable texmf tree to install in. We'll use `%ROOT%` to refer
    to the root directory of that tree. On the "Roots" tab of the MikTeX
    options, you can select any directory, which is not a MikTeX maintained
    root directory. If you don't find a suitable candidate, create a new
    directory and add it to that tab.

 3) Unzip (using 7-zip or winzip or ...) `kulemt-tds.zip` (from the
    [ftp site](ftp://ftp.esat.kuleuven.be/latex/kulemt/)) into the
    `%ROOT%` directory.

 4) If you also want to install the sources, unzip `kulemt-src.zip` (from
    the ftp site) in the directory `%ROOT%\source\latex` .

 5) Update the filename database by clicking the "Refresh FNDB" button on
    the General tab of the MikTeX options.

Note: All documentation is available in the `%ROOT%\doc\latex\kulemt directory`.


Change history
--------------

  * v1.9  (2017-05-30)

      - Pass the `oldfontcommands` to memoir. (kulemt.cls)
      - Update the masters: bwk, cws, ecws, mai, vlit. (kulemt.cfg)
	  - Add master evlit. (kulemt.cfg)

  * v1.8a (2015-06-01)

      - Update Nano options and add some English masters. (kulemt.cfg)

  * v1.8  (2015-05-05)

      - Allow for commands of other packages (e.g., cleveref) to globally
        modify local control sequences such as `\@tempa`. (kulemt.cls)

  * v1.7  (2013-05-01)

      - Logos on front pages completely reworked due to new KU Leuven rules:
        new layout, Sedes removed, only color logos, combined logo of the
        KU Leuven and of the Faculty. (kulemt.cls)
      - Option "bindcover" becomes obsolete because of new cover page layout.
        (kulemt.cls)
      - Option "promoter" is an alias to option "promotor". (kulemt.cls)
      - Input encoding initialized to ASCII to detect other input encodings.
        (kulemt.cls)
      - Master programs can disallow master options. (kulemt.cls)
      - Master option abbreviations are either valid or obsolete.
        Additionally you can specify the list of abbreviations, where one
        must be selected from, if the master requires so. (kulemt.cls)
      - Masters and options updated to the 2013 situation. (kulemt.cfg)
      - Some masters disallow master options. (kulemt.cfg)

  * v1.6b (2012-05-15)

      - Master title of "emtk" corrected. (kulemt.cfg)

  * v1.6  (2012-05-13)

      - New option "bindcover". (kulemt.cls)
      - The environment "abstract*" has a new optional argument to specify
    	the language. (kulemt.cls)
      - "K.U.Leuven" replaced by "KU Leuven". (kulemt.cls)
      - Support for obsolete master options. (kulemt.cls)
      - Missing master options generate a warning instead of an error.
	    (kulemt.cls)
      - Masters and options updated to the 2012 situation. (kulemt.cfg)

  * v1.5  (2011-08-10)

      - `\latinencoding` is T1 only if T1 is the current encoding.
        This resolves a microtype error when using the default CMR fonts.
        (kulemt.cls)

  * v1.4  (2011-06-08)

      - `\mainmatter*` works again. (kulemt.cls)
      - The labels for promoters, assessors and assistants can be defined
        in the configuration file. (kulemt.cls)
      - Support for obsolete masters. (kulemt.cls)
      - New master "mse" which replaces the obsolete "mvt". (kulemt.cfg)
      - The label for promoter is changed to "thesis supervisor" and
        for assistant to "mentor". (kulemt.cfg)

  * v1.3  (2011-05-13)

      - Now compatible with the externalization library of tikz. (kulemt.cls)

  * v1.2  (2010-08-03)

      - Disallow empty values for the promoter keyword. (kulemt.cls)
      - Don't put empty fields on the front page. (kulemt.cls)

  * v1.1  (2010-03-07)

      - Bug fix: Make accented characters work on the front page. (kulemt.cls)

  * v1.0  (2010-03-02)

      - Initial version
