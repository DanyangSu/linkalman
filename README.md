# linkalman
Kalman Filter with flexible form. Work in Progress. Do NOT use yet.  

## Week 1

1. ~~Finalize Model design~~
2. ~~Draft V0~~
3. ~~Derive the math for the whole system~~

## Week 2 

1. ~~Finish Draft 1~~

## Week 3

1. ~~Finish Draft 1 of theory.tex with variable system matrices~~
2. ~~Figure out sequential Update of diagonal R~~
3. ~~LDL decomposition of non-diagonal R~~

## Week 4

1. ~~Implementation Design~~
2. ~~Rewrite derivation of smoother and filter with sequential update~~
3. ~~Add EM section~~

## Week 5

1. ~~Create M read mode, and get it work~~
2. ~~Finish delta2 and chi2~~
3. ~~Read other package codes and figure out where to add Cholesky transform and reverse~~
4. ~~Derive the complete delta2 and chi2~~
5. ~~Masked values~~
6. ~~Rethink the structure of filter and smoother~~

## Week 6

1. ~~Modify smoother section. Add xi2 and xi_t_xi_1t_T~~
2. ~~Finish EM~~
3. ~~Rewrite things in array form~~
4. ~~Finish filter and smoother~~
5. ~~Clean up equations numbers etc~~
6. ~~Finish simple model~~
7. ~~Finish complex ts model~~
8. ~~produce projections~~
9. ~~Rename simple_EM to constantEM~~

## Week 7

1. ~~Test utils~~
2. ~~Test Filter~~
3. ~~Test Smoother~~
4. ~~Test constant_M (need to deal with correct `__setitem__`)~~
5. Test SimpleEM, Constant_EM and CycleEM
6. Test core EM
7. Check wrapped, if change value to 0 (due to missing obs), whether it’s stored. 
8. Finish utility function (plot, etc)
9. Run full test

## Week 8

1. ~~Make it a package~~
2. Review and update documentation
3. Release V1
4. Add test 
5. Add log/doc files

## Week 9+

1. Disturbance smoother (may need general form)
2. Square Root implementation (probably not necessary)
3. Proper Initialization value
4. More General Form (if necessary)
Potential New threads:
1. Non-Gaussian (UKF)
2. Partical Filter
3. IF2 Filter
