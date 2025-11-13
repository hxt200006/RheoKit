c
c
c     ################################################################
c     ##                                                            ##
c     ##  program shear -- pull stress tens. components from output ##
c     ##                                                            ##
c     ################################################################
c
c  Read a list of tinker-hp output files and extract stress tensor data
c  into plain text ".str" files. Each input is reduced to its basename,
c  extension changed from ".out" to ".str", and optionally written into
c  a user-specified directory via the -d / --outdir flag.
c
c Compile: gfortran -03 -o shear shear.f
c
      program shear
      implicit none

      integer        iout,ios,in_unit,list_unit,out_unit
      integer        man_unit
      integer        narg,i,n
      integer        pos
      integer        lastslash
      integer        step
      real*8         etot,epot,ekin,temp,pres
      real*8         sxx,syy,szz,sxy,syz,sxz
      logical        has_outdir
      logical        has_list
      logical        has_man, man_open
      character*240  fname
      character*240  line
      character*240  trimmed
      character*240  outfname
      character*240  arg, val, listfile
      character*240  dir
      character*240  base
      character*240  outdir, manifest
c
c     standard units
c
      iout  = 6
c
c     set default values
c
      has_outdir = .false.
      has_list   = .false.
      has_man    = .false.
      outdir     = ' '
      listfile   = 'HPoutputs.txt'
      manifest   = 'stress_manifest.txt'
c
c     check for arguments 
c
      narg = iargc()
      i = 1
   5  if (i .le. narg) then
         call getarg(i,arg)
c
c     --outdir=dir
c
      if (arg(1:8) .eq. '--outdir') then
            pos = index(arg,'=')
            if (pos .gt. 0) then
               outdir = trim(arg(pos+1:))
               has_outdir = .true.
               if (len_trim(outdir) .gt. 1) then
                  if (outdir(len_trim(outdir):len_trim(outdir)) .eq.
     &               '/') then
                     outdir = outdir(:len_trim(outdir)-1)
                  endif
               endif
            else
               write (iout,*) 'error: use --outdir=path'
            endif
            i = i + 1
            goto 5
c
c     -d dir
c
         else if (arg(1:2) .eq. '-d') then
            if (i+1 .le. narg) then
               call getarg(i+1,outdir)
               outdir = trim(outdir)
               has_outdir = .true.
               if (len_trim(outdir) .gt. 1) then
                  if (outdir(len_trim(outdir):len_trim(outdir)) .eq.
     &               '/') then
                     outdir = outdir(:len_trim(outdir)-1)
                  endif
               endif
               i = i + 2
            else
               write (iout,*) 'error: -d requires a directory path'
               i = i + 1
            endif
            goto 5
c
c     --list=file
c
         else if (arg(1:6) .eq. '--list') then
            pos = index(arg,'=')
            if (pos .gt. 0) then
               listfile = trim(arg(pos+1:))
               has_list = .true.
            else
               write (iout,*) 'error: use --list=file'
            endif
            i = i + 1
            goto 5
c
c     -l file
c
         else if (arg(1:2) .eq. '-l') then
            if (i+1 .le. narg) then
               call getarg(i+1,listfile)
               listfile = trim(listfile)
               has_list = .true.
               i = i + 2
            else
               write (iout,*) 'error: -l requires a file path'
               i = i + 1
            endif
            goto 5
c
c     --manifest=file
c
         else if (arg(1:10) .eq. '--manifest') then
            pos = index(arg,'=')
            if (pos .gt. 0) then
               manifest = trim(arg(pos+1:))
               has_man = .true.
            else
               write (iout,*) 'error: use --manifest=file'
            endif
            i = i + 1
            goto 5
c
c     -m file
c
         else if (arg(1:2) .eq. '-m') then
            if (i+1 .le. narg) then
               call getarg(i+1,manifest)
               manifest = trim(manifest)
               has_man = .true.
               i = i + 2
            else
               write (iout,*) 'error: -m requires a file path'
               i = i + 1
            endif
            goto 5
c
c     -h / --help
c
         else if (arg(1:2) .eq. '-h' .or.
     &            arg(1:6) .eq. '--help') then
            write(iout,*) ' '
            write(iout,*) ' Usage: shear [options]'
            write(iout,*) ' '
            write(iout,*) ' Extract stress tensor data from Tinker-HP'
     &                //  ' output files and write' 
            write(iout,*) ' results to .str files. Creates manifest'
     &                //  ' file with paths to each'
            write(iout,*) ' .str file generated in CWD or in the'
            write(iout,*) ' directory passed to -d --outdir='
            write(iout,*) ' '
            write(iout,*) ' Defaults:'
            write(iout,*) '   --list=HPoutputs.txt'
            write(iout,*) '   --manifest=stress_manifest.txt'
            write(iout,*) ' '
            write(iout,*) ' Options:'
            write(iout,*) '   -d dir  , --outdir=dir      '
     &                 // ' Write .str files to DIR'
            write(iout,*) '   -l file , --list=file       '
     &                 // ' Read FILE with Tinker-HP outputs'
            write(iout,*) '   -m file , --manifest=file   '
     &                 // ' Write .str manifest to FILE'
            write(iout,*) '   -h      , --help            '
     &                 // ' Print this help message'
            write(iout,*) ' '
            stop
c
c     other arguments
c
         else
            write(iout,*) 'warning: ignoring arg: ', trim(arg)
            i = i + 1
            goto 5
         endif
      endif
c
c     end argument parsing
c     open list file
c
      if (has_list) then
         open(newunit=list_unit, file=trim(listfile), status='old',
     &        action='read', iostat=ios)
      else
         write(iout,'(a)', advance='no') ' Enter Name of the List'
     &                                // ' File with .out Entries'
     &                                // ' [HPoutputs.txt]: '
         read(*,'(a)') listfile
         if (len_trim(listfile) .eq. 0) listfile = 'HPoutputs.txt'
         listfile = adjustl(trim(listfile))
         open(newunit=list_unit, file=trim(listfile), status='old',
     &        action='read', iostat=ios)
      endif

      if (ios .ne. 0) then
         write(iout,*) ' Error: could not open ', trim(listfile),
     &                 ' (iostat=', ios, ')'
      stop
      endif
c
c     open stress manifest
c
      if (.not. has_man) then
         manifest = 'stress_manifest.txt'
      endif
      if (has_outdir) then
         manifest = trim(outdir) // '/' // trim(manifest) 
      endif
      man_open = .false.
      open(newunit=man_unit, file=trim(manifest),
     &        status='replace', action='write', iostat=ios)
      if (ios .ne. 0) then 
         write(iout,*) ' Warning: could not create manifest file'
         man_open = .false.
      else
         man_open = .true.
         write(man_unit,'(13x,64a1)') ('#', i=1,64)
         write(man_unit,'(13x,a,60a1,a)') '##', (' ', i=1,60), '##'
         write(man_unit,'(13x,a,a)') '##  RheoKit: Shear         ',
     &                   '                                   ##'

         write(man_unit,'(13x,a,43a1,a)') '##  By Daniel Relix', 
     &                                   (' ', i=1,43), '##'
         write(man_unit,'(13x,a,60a1,a)') '##', (' ', i=1,60), '##'
         write(man_unit,'(13x,64a1)') ('#', i=1,64)
         write(man_unit,*)
      endif
c
c     loop over .out files
c
   10 continue
      read(list_unit, '(a)', iostat=ios) fname
      if (ios .ne. 0) goto 99
      fname = trim(fname)
      n = len_trim(fname)
c
c     get the basename of fname
c
      lastslash = 0
      do i = n, 1, -1
         if (fname(i:i)  .eq. '/') then
            lastslash = i
            exit
         endif
      enddo

      if (lastslash .eq. n) then
         write(iout,*) ' Warning: skip: ', trim(fname)
         goto 10
      endif
      if (lastslash .gt. 0) then
         dir  = fname(1:lastslash-1)
         base = fname(lastslash+1:n)
      else
         dir  = '.'
         base = fname(1:n)
      endif
      
      n = len_trim(base)
      if (n .ge. 4) then
         if (base(n-3:n) .eq. '.out') then
            base = base(:n-4) // '.str'
         endif
      else
         base = trim(base) // '.str'
      endif
      if (has_outdir) then
         outfname = trim(outdir) // '/' // trim(base)
      else
         outfname = trim(dir) // '/' // trim(base)
      endif
      write(iout,*) ' Processing ', trim(outfname)
      if (man_open) then
         if (has_outdir) then
            write(man_unit, '(a)') trim(base)
         else
            write(man_unit, '(a)') trim(outfname)
         endif
      endif
c
c     open the .out file for reading and .str file for writing
c
      open(newunit=in_unit, file=trim(fname), status='old',
     &     action='read', iostat=ios)
      if (ios .eq. 0) then
         open(newunit=out_unit, file=trim(outfname),
     &        status='replace', action='write', iostat=ios)
         if (ios .eq. 0) then
            write(out_unit,'(a8,6a15)') ' MD Step',
     &            'Stress(xx)', 'Stress(yy)', 'Stress(zz)',
     &            'Stress(xy)', 'Stress(yz)', 'Stress(xz)'
c
c     loop over every line of the input .out
c
   20       continue
            read(in_unit,'(a)',iostat=ios) line
            if (ios .ne. 0) goto 30

            trimmed = adjustl(line)

            if (trimmed(1:1) .lt. '0' .or.
     &          trimmed(1:1) .gt. '9') goto 20        

            read(trimmed, *, iostat=ios) step, etot, epot, ekin,
     &           temp, pres, sxx, syy, szz, sxy, syz, sxz
            if (ios .eq. 0) then
               write(out_unit,'(i8,6f15.4)')
     &               step, sxx, syy, szz, sxy, syz, sxz
            endif
            goto 20

   30       continue
            close(in_unit)
            close(out_unit)
         endif
      endif
      goto 10
   99 continue
      if (man_open) then
         close(man_unit)
      endif
      end program shear 
