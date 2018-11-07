import argparse, sys, os
from source.distributor import Distributor

def command_line_parser(args):
	"""Parse command line input
	Recognised arguments:
		-f fileslist (mandatory)
		-n nodeslist (mandatory)
		-o outputfile (optional)

    Args:
        args: command line arguments

	Returns:
		parsed_args: arguments appropriately parsed
    """
	parser = argparse.ArgumentParser(description="Given a list of files with their size, and a list of nodes with their size, produce a distribution plan such that the total amount of data on each node is as even as possible.")
	parser.add_argument("-f", dest="fileslist", required=True,help="List of files with sizes")
	parser.add_argument("-n", dest="nodeslist", required=True,help="List of nodes with sizes")
	parser.add_argument("-o", dest="outputfile", required=False,help="Output file (optional)")
	return parser.parse_args(args)	



def main():

	args = command_line_parser(sys.argv[1:])
	#Read the data into tables
	dist = Distributor(args.fileslist,args.nodeslist,binpackAlgo='emptiest')
	#Propose a distribution plan
	distributionTable = dist.allocate_files_to_nodes()
	distributionPlanString = distributionTable.to_string(columns=['node'],header=False,index_names=False)
	#Print or write to file
	if (args.outputfile):
		with open(args.outputfile,'w') as f:
			f.write(distributionPlanString)
	else:
		print distributionPlanString

if __name__ == "__main__":
	main()