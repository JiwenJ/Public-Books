//#include "CSystem.h"

template <class T>
void CSystem<T>::allocate (unsigned int unRows, unsigned int unCols, T** &aatArray)
{
    aatArray = new T*[unRows];
    aatArray[0] = new T[unRows*unCols];
    for (unsigned int unRow = 1; unRow < unRows; unRow++)
    {
        aatArray[unRow] = &aatArray[0][unCols*unRow];
    }
}

template <class T>
void CSystem<T>::deallocate (T**& aatArray)
{
    delete[] aatArray[0];
    delete[] aatArray;
    aatArray = 0;
}

template <class T>
void CSystem<T>::allocate (unsigned int unRows, unsigned int unCols, unsigned int unSlices, T*** &aaatArray)
{
    aaatArray = new T**[unSlices];
    aaatArray[0] = new T*[unSlices*unCols];
    aaatArray[0][0] = new T[unSlices*unRows*unCols];
    for (unsigned int unSlice = 0; unSlice < unSlices; unSlice++)
    {
        aaatArray[unSlice] = &aaatArray[0][unRows*unSlice];
        for (unsigned int unRow = 0; unRow < unRows; unRow++)
        {
            aaatArray[unSlice][unRow] =
                &aaatArray[0][0][unCols*(unRow+unRows*unSlice)];
        }
    }
}

template <class T>
void CSystem<T>::deallocate (T***& aaatArray)
{
//	fairAssert(aaatArray != NULL, "Assertion while trying to deallocate null pointer reference");
    delete[] aaatArray[0][0];
    delete[] aaatArray[0];
    delete[] aaatArray;
    aaatArray = 0;
}
