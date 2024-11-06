/*
 * File:   DLinkedList.h
 */

#ifndef DLINKEDLIST_H
#define DLINKEDLIST_H

#include <iostream>
#include <sstream>
#include <type_traits>

#include "IList.h"

using namespace std;

template <class T>
class DLinkedList : public IList<T> {
public:
    class Node;        // Forward declaration
    class Iterator;    // Forward declaration
    class BWDIterator; // Forward declaration

protected:
    Node *head; // this node does not contain user's data
    Node *tail; // this node does not contain user's data
    int count;

    bool (*itemEqual)(T &lhs, T &rhs);        // function pointer: test if two items
                                              // (type: T&) are equal or not
    void (*deleteUserData)(DLinkedList<T> *); // function pointer: be called to remove items (if they
                                              // are pointer type)

public:
    DLinkedList(void (*deleteUserData)(DLinkedList<T> *) = 0, bool (*itemEqual)(T &, T &) = 0);

    DLinkedList(const DLinkedList<T> &list);

    DLinkedList<T> &operator=(const DLinkedList<T> &list);

    ~DLinkedList();

    // Inherit from IList: BEGIN
    void add(T e);

    void add(int index, T e);

    T removeAt(int index);

    bool removeItem(T item, void (*removeItemData)(T) = 0);

    bool empty();

    int size();

    void clear();

    T &get(int index);

    int indexOf(T item);

    bool contains(T item);

    string toString(string (*item2str)(T &) = 0);

    // Inherit from IList: END

    void println(string (*item2str)(T &) = 0) {
        cout << toString(item2str) << endl;
    }

    void setDeleteUserDataPtr(void (*deleteUserData)(DLinkedList<T> *) = 0) {
        this->deleteUserData = deleteUserData;
    }

    bool contains(T array[], int size) {
        int idx = 0;
        for(DLinkedList<T>::Iterator it = begin(); it != end(); it++) {
            if(!equals(*it, array[idx++], this->itemEqual)) return false;
        }
        return true;
    }

    /*
     * free(DLinkedList<T> *list):
     *  + to remove user's data (type T, must be a pointer type, e.g.: int*,
     * Point*)
     *  + if users want a DLinkedList removing their data,
     *      he/she must pass "free" to constructor of DLinkedList
     *      Example:
     *      DLinkedList<T> list(&DLinkedList<T>::free);
     */
    static void free(DLinkedList<T> *list) {
        typename DLinkedList<T>::Iterator it = list->begin();
        while(it != list->end()) {
            delete *it;
            it++;
        }
    }

    /* begin, end and Iterator helps user to traverse a list forwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::Iterator it;
     for(it = list.begin(); it != list.end(); it++){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    Iterator begin() {
        return Iterator(this, true);
    }

    Iterator end() {
        return Iterator(this, false);
    }

    /* last, beforeFirst and BWDIterator helps user to traverse a list backwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::BWDIterator it;
     for(it = list.last(); it != list.beforeFirst(); it--){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    BWDIterator bbegin() {
        return BWDIterator(this, true);
    }

    BWDIterator bend() {
        return BWDIterator(this, false);
    }

protected:
    static bool equals(T &lhs, T &rhs, bool (*itemEqual)(T &, T &)) {
        if(&lhs == nullptr || &rhs == nullptr) return false;
        if(itemEqual == 0) return lhs == rhs;
        else return itemEqual(lhs, rhs);
    }

    void copyFrom(const DLinkedList<T> &list);

    void removeInternalData();

    Node *getPreviousNodeOf(int index);

    //////////////////////////////////////////////////////////////////////
    ////////////////////////  INNER CLASSES DEFNITION ////////////////////
    //////////////////////////////////////////////////////////////////////

public:
    class Node {
    public:
        T data;
        Node *next;
        Node *prev;
        friend class DLinkedList<T>;

    public:
        Node(Node *next = 0, Node *prev = 0) {
            this->next = next;
            this->prev = prev;
        }

        Node(T data, Node *next = 0, Node *prev = 0) {
            this->data = data;
            this->next = next;
            this->prev = prev;
        }
    };

    //////////////////////////////////////////////////////////////////////
    class Iterator {
    private:
        DLinkedList<T> *pList;
        Node *pNode;

    public:
        Iterator(DLinkedList<T> *pList = 0, bool begin = true) {
            if(begin) {
                if(pList != 0) this->pNode = pList->head->next;
                else pNode = 0;
            } else {
                if(pList != 0) this->pNode = pList->tail;
                else pNode = 0;
            }
            this->pList = pList;
        }

        Iterator &operator=(const Iterator &iterator) {
            this->pNode = iterator.pNode;
            this->pList = iterator.pList;
            return *this;
        }

        void remove(void (*removeItemData)(T) = 0) {
            pNode->prev->next = pNode->next;
            pNode->next->prev = pNode->prev;
            Node *pNext = pNode->prev; // MUST prev, so iterator++ will go to end
            if(removeItemData != 0) removeItemData(pNode->data);
            delete pNode;
            pNode = pNext;
            pList->count -= 1;
        }

        T &operator*() {
            return pNode->data;
        }

        bool operator!=(const Iterator &iterator) {
            return pNode != iterator.pNode;
        }

        // Prefix ++ overload
        Iterator &operator++() {
            pNode = pNode->next;
            return *this;
        }

        // Postfix ++ overload
        Iterator operator++(int) {
            Iterator iterator = *this;
            ++*this;
            return iterator;
        }
    };

    class BWDIterator {
    private:
        DLinkedList<T> *pList;
        Node *pNode;

    public:
        BWDIterator(DLinkedList<T> *pList = 0, bool begin = true) {
            if(begin) {
                if(pList != 0) this->pNode = pList->tail->prev;
                else pNode = 0;
            } else {
                if(pList != 0) this->pNode = pList->head;
                else pNode = 0;
            }
            this->pList = pList;
        }

        // Assignment operator
        BWDIterator &operator=(const BWDIterator &iterator) {
            this->pNode = iterator.pNode;
            this->pList = iterator.pList;
            return *this;
        }

        // (a) Operator !=
        bool operator!=(const BWDIterator &iterator) {
            return pNode != iterator.pNode;
        }

        // (b) Operator ++ (prefix)
        BWDIterator &operator++() {
            if(pNode) {
                pNode = pNode->prev; // Move backward in the list
            }
            return *this;
        }

        // (b) Operator ++ (postfix)
        BWDIterator operator++(int) {
            BWDIterator iterator = *this;
            ++*this;
            return iterator;
        }

        // (c) Operator *
        T &operator*() {
            return pNode->data;
        }
    };
};

//////////////////////////////////////////////////////////////////////
// Define a shorter name for DLinkedList:

template <class T>
using List = DLinkedList<T>;

//////////////////////////////////////////////////////////////////////
////////////////////////     METHOD DEFNITION      ///////////////////
//////////////////////////////////////////////////////////////////////

template <class T>
DLinkedList<T>::DLinkedList(void (*deleteUserData)(DLinkedList<T> *), bool (*itemEqual)(T &, T &)) {
    this->deleteUserData = deleteUserData;
    this->itemEqual = itemEqual;

    head = new Node();
    tail = new Node();

    head->next = tail;
    tail->prev = head;

    count = 0;
}

template <class T>
DLinkedList<T>::DLinkedList(const DLinkedList<T> &list) {
    this->deleteUserData = list.deleteUserData;
    this->itemEqual = list.itemEqual;

    head = new Node();
    tail = new Node();

    head->next = tail;
    tail->prev = head;

    count = 0;

    Node *current = list.head->next;
    while(current != list.tail) {
        add(current->data); // Hàm add phải sao chép đúng đối tượng
        current = current->next;
    }
}

template <class T>
DLinkedList<T> &DLinkedList<T>::operator=(const DLinkedList<T> &list) {
    if(this != &list) {

        clear();

        this->deleteUserData = list.deleteUserData;
        this->itemEqual = list.itemEqual;

        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->prev = head;
        count = 0;

        Node *current = list.head->next;
        while(current != list.tail) {
            add(current->data);
            current = current->next;
        }
    }
    return *this;
}

template <class T>
DLinkedList<T>::~DLinkedList() {
    clear();
    delete head;
    delete tail;
}

template <class T>
void DLinkedList<T>::add(T e) {
    // TODO
    Node *newNode = new Node(e);

    newNode->prev = tail->prev;
    newNode->next = tail;
    tail->prev->next = newNode;
    tail->prev = newNode;
    count++;
}

template <class T>
void DLinkedList<T>::add(int index, T e) {
    if(index < 0 || index > count) {
        throw out_of_range("Index is out of range!");
    }

    Node *newNode = new Node(e);

    if(index == count) {
        newNode->prev = tail->prev;
        newNode->next = tail;
        tail->prev->next = newNode;
        tail->prev = newNode;
    } else {
        Node *current = getPreviousNodeOf(index);
        newNode->next = current->next;
        newNode->prev = current;

        if(current->next != nullptr) {
            current->next->prev = newNode;
        }
        current->next = newNode;
    }

    count++;
}

template <class T>
typename DLinkedList<T>::Node *DLinkedList<T>::getPreviousNodeOf(int index) {
    if(index < 0 || index > count) {
        throw out_of_range("Index is out of range!");
    }

    Node *current = head;
    for(int i = 0; i < index; ++i) {
        current = current->next;
    }

    return current;
}

template <class T>
T DLinkedList<T>::removeAt(int index) {
    if(index < 0 || index >= count) {
        throw out_of_range("Index is out of range!");
    }

    Node *current = getPreviousNodeOf(index + 1);
    T data = current->data;

    if(current->prev != nullptr) {
        current->prev->next = current->next;
    }
    if(current->next != nullptr) {
        current->next->prev = current->prev;
    }

    delete current;

    count--;

    return data;
}

template <class T>
bool DLinkedList<T>::empty() {
    // TODO
    return count == 0;
}

template <class T>
int DLinkedList<T>::size() {
    // TODO
    return count;
}

template <class T>
void DLinkedList<T>::clear() {
    if(head == nullptr || tail == nullptr) {
        return;
    }

    Node *current = head->next;

    while(current != tail) {
        Node *nextNode = current->next;
        delete current;
        current = nextNode;
    }

    head->next = tail;
    tail->prev = head;
    count = 0;
}

template <class T>
T &DLinkedList<T>::get(int index) {
    if(index < 0 || index >= count) {
        throw out_of_range("Index is out of range!");
    }

    Node *current = getPreviousNodeOf(index + 1);
    return current->data;
}

template <typename T>
int DLinkedList<T>::indexOf(T value) {

    if(head == nullptr) {
        return -1;
    }

    Node *current = head->next;
    int index = 0;

    while(current != tail) {
        if(equals(current->data, value, this->itemEqual)) {
            return index;
        }
        current = current->next;
        index++;
    }

    return -1;
}

template <class T>
bool DLinkedList<T>::removeItem(T item, void (*removeItemData)(T)) {
    Iterator it = begin();

    while(it != end()) {
        if(equals(*it, item, this->itemEqual)) {
            it.remove(removeItemData);
            return true;
        }
        ++it;
    }

    return false;
}

template <class T>
bool DLinkedList<T>::contains(T item) {
    Iterator it = begin();

    while(it != end()) {
        if(equals(*it, item, this->itemEqual)) {
            return true;
        }
        ++it;
    }

    return false;
}

template <class T>
string DLinkedList<T>::toString(string (*item2str)(T &)) {
    if(head == nullptr || tail == nullptr) {
        return "[]";
    }

    stringstream ss;
    Node *current = head->next;

    ss << "[";
    while(current != tail) {
        if(item2str != nullptr) {
            ss << item2str(current->data);
        } else {
            ss << current->data;
        }

        current = current->next;
        if(current != tail) {
            ss << ", ";
        }
    }
    ss << "]";

    return ss.str();
}

template <class T>
void DLinkedList<T>::copyFrom(const DLinkedList<T> &list) {
    clear();

    if(list.empty()) {
        return;
    }

    for(Iterator it = list.begin(); it != list.end(); ++it) {
        add(*it);
    }
}

template <class T>
void DLinkedList<T>::removeInternalData() {
    Node *current = head->next;
    while(current != tail) {
        Node *nextNode = current->next;
        if(deleteUserData) {
            deleteUserData(current);
        }
        delete current;
        current = nextNode;
    }
    head->next = tail;
    tail->prev = head;
    count = 0;
}

#endif /* DLINKEDLIST_H */
