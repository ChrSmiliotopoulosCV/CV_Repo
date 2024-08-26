# This is .v1 of my newly created python parser, named python_Evtx_Parser(v1 - Beta) for .evtx files. With this
# portable and versatile chunk of code, the entries of Windows Event Logger and Sysmon .evtx files could be
# enumerated to reveal the existence or not of possible Lateral Movement Attacks over Small Office Home Office (SOHO)
# Networks. Along with .v1 of our python script, the parser is going to be initialized under the hood to identify the
# possibility of malicious Lateral Movement Attacks. All the associated filters are based on previous work done on
# the Sysmon config.xml file custom rules. What is special with this beta version of the python .evtx files parser is
# its independence from operating system platforms, namely Windows, macOS and any distribution of Linux OS. This
# would be analyzed thoroughly to the relevant README file which will accompany the distributed .py script on GitHub.

# The source code and supporting material of python_Evtx_Parser(v1 - Beta) is available on
# https://github.com/ChristosSmiliotopoulos/pythonParser.git private repository.

# Importing necessary python libraries

import mmap  # Python's memory-mapped file input and output (I/O) library.

import argparse  # argparse library is a parser for extra command-line options, arguments and sub-commands. This will
# make our python_Evtx_Parser(v1 - Beta) capable to be executed on any Windows cmd or powershell, macOS/Linux
# terminal environment.

from xml.dom import minidom  # Python's compact implementation of the DOM interface enables programmers to parse XML
# files via xml.dom.minidom xml parser.

from evtx.Evtx import FileHeader  # Python library for parsing Windows Event Log files (.evtx). Fileheader() function
# allows .evtx parsing based on the log file headers.

import evtx.Views  # Evtx library's sub-library which renders the given record into an XML document.

from xml.etree import ElementTree as xee  # ElementTree library allows a developer to parse and navigate an .xml,
# providing clever tree-based structure and giving him great insight to work with.

# main() python function named python_Evtx_Parser(). This is the main block of code for the parser which only runs
# when called with PyCharm of any known console, terminal, cmd or powershell. All the necessary parameters will be
# passed to python_Evtx_Parser() as arguments.

def python_Evtx_Parser():
    parser = argparse.ArgumentParser(prog="python_Evtx_Parser", description="Enumeration of .evtx log files based on "
                                                                            "EventID and Lateral Movement Attacks "
                                                                            "oriented filtering.")
    parser.add_argument("-f", "--iFile", dest="ifile", type=str, required=True, help="path to the input file")
    parser.add_argument("-i", "--evtId", dest="id", type=str, default="all", help="id of the Event to Dump")
    parser.add_argument("-o", "--oFile", dest="ofile", type=str, required=False, help="path to the output file")

    args = parser.parse_args()
    outFile = False
    if args.ofile is not None:
        outFile = open(args.ofile, 'a+')

    with open(args.ifile, 'r') as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        fh = FileHeader(buf, 0x00)
        hOut = "<?xml version='1.0' encoding='utf-8' standalone='yes' ?><Events>"
        if outFile:
            outFile.write(hOut)
        else:
            print(hOut)

        Counter = 0

        for strxml, recorn in evtx.Views.evtx_file_xml_view(fh):
            xmlDoc = minidom.parseString(strxml.replace("\n", ""))
            evtId = xmlDoc.getElementsByTagName("EventID")[0].childNodes[0].nodeValue
            if args.id == 'all':
                if outFile:
                    outFile.write(xmlDoc.toprettyxml())
                else:
                    print(xmlDoc.toprettyxml())
            else:
                if evtId == args.id:
                    if outFile:
                        outFile.write(xmlDoc.toprettyxml())
                    else:
                        print(xmlDoc.toprettyxml())

            # ======Trial=====
            doc = xee.fromstring(strxml)

            for tag in doc.findall('Name'):
                if tag.attrib['Image'] == 'lsass':
                    doc.remove(tag)
            stringXMLObject = xee.tostring(doc)
            print(xee.tostring(doc))
            countVar = strxml.count("lsass")
            print(countVar)

            if countVar == 1:
                Counter += 1
            elif countVar == 0:
                Counter = Counter
        print(Counter)
        print("ATTENTION!!!", " ", Counter, " ",
              "Events have been identified on the targetted system as suspicious for Lateral Movement Attack!!!")

        # xmlDoc = minidom.parseString(doc)

        # ===end of Trial===

        buf.close()

        print("=========")

        endTag = "</Events>"

        if outFile:
            outFile.write(endTag)
        else:
            print(endTag)


if __name__ == '__main__':
    python_Evtx_Parser()
