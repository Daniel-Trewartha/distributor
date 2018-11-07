import sys
import pandas as pd

#To do: 
#Clean up comments, do them systematically
#Implement modified first fit algorithm
#Write a good tester

class Distributor:
    """Given a list of files with their sizes and a list of nodes with their sizes, Distributor proposes an distribution plan such that the maximal number of files are allocated and nodes are as balanced as possible
    """ 


    def __init__(self,filesList,nodesList,binpackAlgo='emptiest',subsetAlgo='greedy'):

        """Initialise, read the input data, check its validity and construct appropriate tables

        Args:
            filesList (str): File containing the list of files with sizes. File should contain one file per line, filename and then size separated by a space.
            nodesList (str): File containing the list of nodes with sizes. File should contain one node per line, nodename and then size separated by a space.
            binPackAlgo (str): Algorithm to use for bin packing.
            subsetAlgo (str): Algorithm to use for k subset division

        """

        self.binpackAlgorithms = {'first_fit':self._first_fit_bin_pack,'last_fit':self._last_fit_bin_pack,'emptiest':self._emptiest_fit_bin_pack, 'fullest':self._fullest_fit_bin_pack}
        self.kSubsetAlgorithms = {'greedy':self._greedy_k_subset}

        try:
            self._bin_pack = self.binpackAlgorithms[binpackAlgo]
        except KeyError:
            print("Invalid bin packing algorithm: "+binpackAlgo)
            print("Options are: "+str(self.binpackAlgorithms.keys()))
            sys.exit(1)

        try:
            self._k_subsets = self.kSubsetAlgorithms[subsetAlgo]
        except KeyError:
            print("Invalid bin packing algorithm: "+subsetAlgo)
            print("Options are: "+str(self.kSubsetAlgorithms.keys()))
            sys.exit(1)

        #Read the input data
        try:
            self.filesTable = pd.read_csv(filesList,sep=' ',comment='#',header=None,names=['filename','size'],dtype={'filename':str,'size': int})
            #Add a column for assignment to nodes
            self.filesTable['node'] = 'NULL'

            self.nodesTable = pd.read_csv(nodesList,sep=' ',comment='#',header=None,names=['nodename','totalSpace'],dtype={'nodename':str,'totalSpace': int})
            #Add columns to track allocated space and remaining space on nodes - saves repeat calculations on the fly, and number of nodes should (hopefully!) be small wrt memory
            self.nodesTable['freeSpace'] = self.nodesTable['totalSpace']
            self.nodesTable['allocatedSpace'] = 0

        except Exception as e:
            print("Error parsing input: "+str(e))
            sys.exit(1)

        valid, message = self._check_input_validity()

        if not valid:
            print(message)
            sys.exit(1)

        self.filesTable.set_index('filename',inplace=True)
        self.nodesTable.set_index('nodename',inplace=True)


    def allocate_files_to_nodes(self):
        """Propose a distribution of files to nodes.

        """


        #Consider two cases: 
        #
        #a) The total size of files is large w.r.t. the available space on nodes
        #Nodes should be full, so pack files onto nodes as tightly as possible without regard for balancing.
        #
        #b) Otherwise
        #Divide files into equal subsets, allocate to one subset to each node, then distribute the remainder as evenly as possible.

        fileSizePerNode = self.filesTable['size'].sum()/float( len(self.nodesTable.index) )
        #Compare to median node space to better take account of cases with highly heterogenuous node sizes
        medianNodeSpace = self.nodesTable['totalSpace'].median()

        self.filesTable.sort_values('size',ascending=True,inplace=True)
        self.nodesTable.sort_values('totalSpace',ascending=True,inplace=True)

        if (fileSizePerNode > medianNodeSpace):
            self._bin_pack(self.filesTable,'size',self.nodesTable,'allocatedSpace','freeSpace',self._assign_file_to_node)
        else:
            #Add a column for partitioning into subsets
            self.filesTable['subset'] = None
            #Create one more subset than there are nodes - otherwise half of the nodes will be too small to fit a subset.
            nSubsets = len(self.nodesTable.index)+1
            self.subsetsTable = pd.DataFrame(index=range(nSubsets),columns=['size'])
            self.subsetsTable['size'] = 0
            #Populate the subsets table
            self._k_subsets(self.filesTable,'size',self.subsetsTable,'size',self._assign_file_to_subset)
            self.subsetsTable['assignedToNode'] = False

            self._pack_subsets_onto_nodes()

            #Now find the remaining files that have not yet been allocated to a node and try to do so
            unallocated_files = self.filesTable[self.filesTable['node'] == 'NULL']
            self._bin_pack(unallocated_files,'size',self.nodesTable,'allocatedSpace','freeSpace',self._assign_file_to_node)

        #Check that our proposed distribution makes sense
        if (not self._check_distribution_validity()):           
            print("Allocation algorithm returned invalid distribution plan")
            sys.exit(1)

        return self.filesTable

    def _check_input_validity(self):
        """Check the validity of the files and nodes lists. File sizes and node spaces should be positive, and names should be unique.

        """
        
        valid = True
        message = ""

        if not (self.filesTable['size'] > 0).all():
            message = 'Negative file sizes in input. Input corrupted?'
            valid = False
        if not (self.nodesTable['totalSpace'] > 0).all():
            message = 'Negative node space in input. Input corrupted?'
            valid = False
        
        if not (len(self.filesTable['filename'].value_counts()) == len(self.filesTable.index)):
            message = "Duplicate filenames in input \n"+str(self.filesTable[self.filesTable.duplicated(['filename'],keep=False)]['filename'])
            valid = False
        if not (len(self.nodesTable['nodename'].value_counts()) == len(self.nodesTable.index)):
            message = "Duplicate node names in input \n"+str(self.nodesTable[self.nodesTable.duplicated(['nodename'],keep=False)]['nodename'])
            valid = False

        return valid,message


    def _check_distribution_validity(self):
        """Once files are assigned to nodes, check that the proposed distribution plan fits the constraints

        """
        #Check that all nodes actually exist - should be impossible for this to fail but can't hurt to check.
        for node in self.filesTable.node.unique():
            if node != 'NULL':
                if not node in self.nodesTable.index.values:
                    return False

        #Check we haven't overallocated to any nodes
        allocatedSizes = self.filesTable.groupby('node')[['size']].sum()
        allocatedSizes = allocatedSizes.merge(self.nodesTable,left_index=True,right_index=True)
        if (allocatedSizes['size'] > allocatedSizes.totalSpace).any():
            return False

        return True


    def _pack_subsets_onto_nodes(self):
        """Assign a subset to each node

        """

        for node in range(len(self.nodesTable)):
            largestFit = self.subsetsTable[(~self.subsetsTable['assignedToNode']) & (self.subsetsTable['size'] <= self.nodesTable.loc[self.nodesTable.index[node],'freeSpace'])].head(1)
            if (len(largestFit == 1)): #Check that we have a subset that fits
                self._assign_subset_to_node(largestFit.index[0],self.nodesTable.index[node])


    """Algorithms for bin packing

    """

    def _first_fit_bin_pack(self,values,valuesSizeCol,bins,binsAllocatedCol,binsRemainingCol,assign_function):
        """Pack values into bins using the first fit algorithm.
        Traverse the list of values, assigning each to the first bin that has space for it.

        Args:
            values (pd.dataframe): Dataframe of values with their sizes.  Assumed sorted ascending by size.
            valuesSizeCol (str): The column containing the size of values.
            bins (pd.dataframe): Dataframe of bins with their remaining sizes. Assumed sorted ascending by size.
            binsAllocatedCol (str): The column containing the total amount currently allocated to bins. This column should be appropriately updated as values are assigned to bins.
            binsRemainingCol (str): The column containing the Remaining capacity of bins. This column should be appropriately updated as values are assigned to bins.
            assign_function (:function: value, bin): A function which assigns value to bin, and updates the tables accordingly.

        """
        for v in range(len(values)):
            firstFit = bins[(bins[binsRemainingCol] >= values.iloc[v][valuesSizeCol])].head(1)
            if (len(firstFit) == 1):
                assign_function(values.index[v],firstFit.index[0])

    def _last_fit_bin_pack(self,values,valuesSizeCol,bins,binsAllocatedCol,binsRemainingCol,assign_function):
        """Pack values into bins using the last fit algorithm.
        Traverse the list of values, assigning each to the last bin that has space for it.

        Args:
            values (pd.dataframe): Dataframe of values with their sizes.  Assumed sorted ascending by size.
            valuesSizeCol (str): The column containing the size of values.
            bins (pd.dataframe): Dataframe of bins with their remaining sizes. Assumed sorted ascending by size.
            binsAllocatedCol (str): The column containing the total amount currently allocated to bins. This column should be appropriately updated as values are assigned to bins.
            binsRemainingCol (str): The column containing the Remaining capacity of bins. This column should be appropriately updated as values are assigned to bins.
            assign_function (:function: value, bin): A function which assigns value to bin, and updates the tables accordingly.

        """
        for v in range(len(values)):
            firstFit = bins[(bins[binsRemainingCol] >= values.iloc[v][valuesSizeCol])].tail(1)
            if (len(firstFit) == 1):
                assign_function(values.index[v],firstFit.index[0])

    def _emptiest_fit_bin_pack(self,values,valuesSizeCol,bins,binsAllocatedCol,binsRemainingCol,assign_function):
        """Pack values into bins using the emptiest fit algorithm.
        Traverse the list of values, assigning each to the emptiest bin (ie the one with the least currently assigned to it) that has room for it.

        Args:
            values (pd.dataframe): Dataframe of values with their sizes.  Assumed sorted ascending by size.
            valuesSizeCol (str): The column containing the size of values.
            bins (pd.dataframe): Dataframe of bins with their remaining sizes. Assumed sorted ascending by size.
            binsAllocatedCol (str): The column containing the total amount currently allocated to bins. This column should be appropriately updated as values are assigned to bins.
            binsRemainingCol (str): The column containing the Remaining capacity of bins. This column should be appropriately updated as values are assigned to bins.
            assign_function (:function: value, bin): A function which assigns value to bin, and updates the tables accordingly.

        """
        for v in range(len(values)):
            remainingBins = bins[(bins[binsRemainingCol] >= values.iloc[v][valuesSizeCol])][binsAllocatedCol]
            if(len(remainingBins) > 0):
                firstFit = remainingBins.idxmin(axis=1)
                assign_function(values.index[v],firstFit)

    def _fullest_fit_bin_pack(self,values,valuesSizeCol,bins,binsAllocatedCol,binsRemainingCol,assign_function):
        """Pack values into bins using the fullest fit algorithm.
        Traverse the list of values, assigning each to the fullest bin (ie the one with the most currently assigned to it) that has room for it.

        Args:
            values (pd.dataframe): Dataframe of values with their sizes. Assumed sorted ascending by size.
            valuesSizeCol (str): The column containing the size of values.
            bins (pd.dataframe): Dataframe of bins with their remaining sizes. Assumed sorted ascending by size.
            binsAllocatedCol (str): The column containing the total amount currently allocated to bins. This column should be appropriately updated as values are assigned to bins.
            binsRemainingCol (str): The column containing the Remaining capacity of bins. This column should be appropriately updated as values are assigned to bins.
            assign_function (:function: value, bin): A function which assigns value to bin, and updates the tables accordingly.

        """
        for v in range(len(values)):
            remainingBins = bins[(bins[binsRemainingCol] >= values.iloc[v][valuesSizeCol])][binsAllocatedCol]
            if(len(remainingBins) > 0):
                firstFit = remainingBins.idxmax(axis=1)
                assign_function(values.index[v],firstFit)



    """Algorithsm for k subset division

    """

    def _greedy_k_subset(self,values,valuesSizeCol,subsetsTable,subsetSizeCol,assign_function):
        """Assign values to subsets as evenly as possible using a greedy algorithm.
        Traverse the list of values from largest to smallest, assigning each to the currently emptiest subset.

        Args:
            values (pd.dataframe): Dataframe of values with their sizes.  Assumed sorted ascending by size.
            valuesSizeCol (str): The column containing the size of values.
            subsets (pd.dataframe): Dataframe of subsets with their current sizes.
            binsSizeCol (str): The column containing the size of subsets. This column should be appropriately updated as values are assigned to subsets.
            assign_function (:function: value, bin): A function which assigns value to subset, and updates the tables accordingly.

        """
        for i in reversed(range(len(values))):
            emptiest = subsetsTable[subsetSizeCol].idxmin()
            assign_function(values.index[i],subsetsTable.index[emptiest])

        #Largest subsets first
        subsetsTable.sort_values('size',ascending=False,inplace=True)

    """Functions for assigning an item in one table to an item in another table

    """

    def _assign_subset_to_node(self,subset,nodename):
        """Assign a subset to a node.

        Args:
            subset (str): index of the subset to assign
            nodename (str): index of the node to be assigned to

        """
        self.subsetsTable.loc[subset,'assignedToNode'] = True
        #Assign each file in this subset.
        self.filesTable.apply(lambda row: self._assign_file_to_node(row.name,nodename) if row.subset == subset else row, axis=1)

    def _assign_file_to_node(self,filename, nodename):
        """Assign a file to a node.

        Args:
            file (str): index of the file to assign
            nodename (str): index of the node to be assigned to

        """
        if (self.filesTable.loc[filename,'node'] == 'NULL'):
            self.filesTable.loc[filename,'node'] = nodename
            self.nodesTable.loc[nodename,'freeSpace'] -= self.filesTable.loc[filename,'size']
            self.nodesTable.loc[nodename,'allocatedSpace'] += self.filesTable.loc[filename,'size']

    def _assign_file_to_subset(self,filename,subset):
        """Assign a file to a subset.

        Args:
            file (str): index of the file to assign
            subset(str): index of the subset to be assigned to

        """
        
        self.filesTable.loc[filename,'subset'] = subset
        self.subsetsTable.loc[subset,'size'] += self.filesTable.loc[filename,'size']