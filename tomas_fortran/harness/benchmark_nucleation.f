C     **************************************************
C     *  Nucleation Parameterization Benchmark         *
C     **************************************************
C
C     Standalone benchmark for ricco_nucl and dunne_inorg_nucl.
C     No dependency on sizecode.COM.
C
C     Output:
C       output/nucleation_ricco.csv
C       output/nucleation_dunne.csv

      PROGRAM benchmark_nucleation

      IMPLICIT NONE

C-----VARIABLE DECLARATIONS-------------------------------------------
      integer i, nricco, ndunne
      parameter(nricco=10, ndunne=10)

C     Riccobono test case arrays
      double precision r_temp(nricco), r_h2so4(nricco), r_org(nricco)
      double precision r_fn, r_rnuc

C     Dunne test case arrays
      double precision d_temp(ndunne), d_fion(ndunne)
      double precision d_h2so4(ndunne), d_nh3(ndunne)
      double precision d_Mair(ndunne)
      double precision d_fn, d_rnuc
      double precision d_Jbn, d_Jtn, d_Jbi, d_Jti

C-----RICCOBONO TEST CASES---------------------------------------------
C     Vary T (240-310K), h2so4 (1e5-1e9), org (1e5-1e9)

      data r_temp  / 240.d0, 260.d0, 278.d0, 290.d0, 310.d0,
     &               250.d0, 270.d0, 285.d0, 300.d0, 278.d0 /
      data r_h2so4 / 1.d5,   1.d6,   1.d7,   1.d8,   1.d9,
     &               5.d5,   5.d6,   5.d7,   5.d8,   1.d7   /
      data r_org   / 1.d7,   1.d7,   1.d7,   1.d7,   1.d7,
     &               1.d5,   1.d6,   1.d8,   1.d9,   5.d7   /

C-----DUNNE TEST CASES-------------------------------------------------
C     Vary T (240-310K), h2so4 (1e5-1e9), nh3 (0, 1e7, 1e10),
C     fion (0, 3, 10), Mair=2.5e19

      data d_temp  / 240.d0, 260.d0, 278.d0, 290.d0, 310.d0,
     &               250.d0, 270.d0, 285.d0, 300.d0, 278.d0 /
      data d_fion  / 3.d0,   3.d0,   3.d0,   3.d0,   3.d0,
     &               0.d0,   10.d0,  3.d0,   3.d0,   3.d0   /
      data d_h2so4 / 1.d5,   1.d6,   1.d7,   1.d8,   1.d9,
     &               5.d5,   5.d6,   5.d7,   5.d8,   1.d7   /
      data d_nh3   / 1.d7,   1.d7,   1.d7,   1.d7,   1.d7,
     &               0.d0,   1.d10,  1.d9,   1.d8,   5.d7   /
      data d_Mair  / 2.5d19, 2.5d19, 2.5d19, 2.5d19, 2.5d19,
     &               2.5d19, 2.5d19, 2.5d19, 2.5d19, 2.5d19 /

C-----CODE--------------------------------------------------------------

      write(*,*) '========================================'
      write(*,*) 'Nucleation Parameterization Benchmark'
      write(*,*) '========================================'

C     --- Riccobono benchmark ---
      write(*,*) 'Running Riccobono 2014 test cases...'
      open(unit=10, file='output/nucleation_ricco.csv',
     &     status='replace')
      write(10,'(A)') 'temp,h2so4,org,fn'

      do i=1,nricco
         call ricco_nucl(r_temp(i), r_h2so4(i), r_org(i),
     &                   r_fn, r_rnuc)
         write(10,'(E25.16,A,E25.16,A,E25.16,A,E25.16)')
     &        r_temp(i), ',', r_h2so4(i), ',',
     &        r_org(i), ',', r_fn
         write(*,'(A,I2,A,E12.4,A,E12.4,A,E12.4,A,E12.4)')
     &        '  Case ', i, ': T=', r_temp(i),
     &        ' H2SO4=', r_h2so4(i),
     &        ' org=', r_org(i), ' fn=', r_fn
      enddo
      close(10)
      write(*,*) 'Wrote output/nucleation_ricco.csv'

C     --- Dunne benchmark ---
      write(*,*) ''
      write(*,*) 'Running Dunne 2016 test cases...'
      open(unit=10, file='output/nucleation_dunne.csv',
     &     status='replace')
      write(10,'(A)')
     &     'temp,fion,h2so4,nh3,Mair,fn,Jbn,Jtn,Jbi,Jti'

      do i=1,ndunne
         call dunne_inorg_nucl(d_temp(i), d_fion(i), d_h2so4(i),
     &        d_nh3(i), d_Mair(i), d_fn, d_rnuc,
     &        d_Jbn, d_Jtn, d_Jbi, d_Jti)
         write(10,'(E25.16,A,E25.16,A,E25.16,A,E25.16,A,E25.16,
     &        A,E25.16,A,E25.16,A,E25.16,A,E25.16,A,E25.16)')
     &        d_temp(i), ',', d_fion(i), ',',
     &        d_h2so4(i), ',', d_nh3(i), ',',
     &        d_Mair(i), ',', d_fn, ',',
     &        d_Jbn, ',', d_Jtn, ',',
     &        d_Jbi, ',', d_Jti
         write(*,'(A,I2,A,E12.4,A,E12.4,A,E12.4,A,E12.4,A,E12.4)')
     &        '  Case ', i, ': T=', d_temp(i),
     &        ' H2SO4=', d_h2so4(i),
     &        ' nh3=', d_nh3(i), ' fion=', d_fion(i),
     &        ' fn=', d_fn
      enddo
      close(10)
      write(*,*) 'Wrote output/nucleation_dunne.csv'

      write(*,*) ''
      write(*,*) '========================================'
      write(*,*) 'Nucleation benchmark complete!'
      write(*,*) '========================================'

      END PROGRAM
