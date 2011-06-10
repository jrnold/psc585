Assignment4
==============


This is my write-up for :download:`Assignment 4 <pdf/PSC585Ex4.pdf>`

Most of the code to implement this assignment is in
:py:class:`psc585.FinalModel`.  The attributes of this class includes
the data from ``FinalModel.mat`` and ``FinalData.mat`` provided with
the assignment.  

The method :py:func:`psc585.FinalModel.new_p` implements ``NewP``; 
:py:func:`psc585.FinalModel.phigprov` implements ``Phihprov``; 
:py:func:`psc585.FinalModel.ptilde` implements ``Ptilde``.

The Nested Pseudo Likelihood estimator described in part (d) is
implemented in the method :py:func:`psc585.FinalModel.npl`.

The maximization of the partial pseudo-likelihood estimator in part
(e) is implemented in the method :py:func:`psc585.argmax_theta`.


