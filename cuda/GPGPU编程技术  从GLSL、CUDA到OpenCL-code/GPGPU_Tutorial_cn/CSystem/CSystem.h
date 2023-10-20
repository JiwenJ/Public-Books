#ifndef CSYSTEM_H_
#define CSYSTEM_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
/**
 * @class CSystem
 * @brief This class encapsulates system specific calls
 * @author Stefan May
 * @update Deyuan Qiu
 */
template <class T>
class CSystem
{
	public:
    /**
     * Allocation of 2D arrays
     * @param unRows number of rows
     * @param unCols number of columns
     * @param aatArray data array
     */
    static void allocate (unsigned int unRows, unsigned int unCols, T** &aatArray);
    /**
     * Deallocation of 2D arrays. Pointers are set to null.
     * @param aatArray data array
     */
    static void deallocate (T** &aatArray);
    /**
     * Allocation of 3D arrays
     * @param unRows number of rows
     * @param unCols number of columns
     * @param unSlices number of slices
     * @param aaatArray data array
     */
    static void allocate (unsigned int unRows, unsigned int unCols, unsigned int unSlices, T*** &aaatArray);
    /**
     * Deallocation of 3D arrays. Pointers are set to null.
     * @param aaatArray data array
     */
    static void deallocate (T*** &aaatArray);
};

#include "CSystem.cpp"
#endif /*CSYSTEM_H_*/
