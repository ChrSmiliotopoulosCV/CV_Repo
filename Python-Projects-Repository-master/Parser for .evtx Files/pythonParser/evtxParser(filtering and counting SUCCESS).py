import mmap
import argparse
from xml.dom import minidom

from evtx.Evtx import FileHeader
import evtx.Views

from xml.etree import ElementTree as xee

import lxml.etree
import pprint


def main():
    parser = argparse.ArgumentParser(prog="evtIdDumper", description="Specify eventID to dump")
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
        print("ATTENTION!!!", " ", Counter, " ", "Events have been identified on the targetted system as suspicious for Lateral Movement Attack!!!")





            # xmlDoc = minidom.parseString(doc)

            # ===end of Trial===

            # evtId = xmlDoc.getElementsByTagName("EventID")[0].childNodes[0].nodeValue
            # if args.id == 'all':
            #     if outFile:
            #         outFile.write(xmlDoc.toprettyxml())
            #     else:
            #         print(xmlDoc.toprettyxml())
            # else:
            #     if evtId == args.id:
            #         if outFile:
            #             outFile.write(xmlDoc.toprettyxml())
            #         else:
            #             print(xmlDoc.toprettyxml())
        buf.close()

        print("=========")

        endTag = "</Events>"

        if outFile:
            outFile.write(endTag)
        else:
            print(endTag)

if __name__ == '__main__':
    main()


